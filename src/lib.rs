mod dtype;
pub use dtype::DType;
pub mod mic;
pub mod models;
pub(crate) mod utils;

use std::{
    cmp::Ordering::{self, Equal},
    fmt::Debug,
    mem,
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
};

use cpal::{
    traits::{DeviceTrait, HostTrait},
    SampleRate, SupportedStreamConfigRange,
};
use mic::Settings;
use models::{CommonModelParams, Model, ModelDefinition};
use thingbuf::recycling::WithCapacity;
use thiserror::Error;
use tracing::{error, info, instrument, warn, Level};

use tokio::sync::{mpsc, oneshot};

macro_rules! parse_data {
    ($t:ty, $device:ident, $config:ident, $tx: ident, $msl: ident) => {{
        use cpal::traits::{DeviceTrait, StreamTrait};
        use dasp_frame::Frame;
        use dasp_signal::Signal;
        use tracing::error;

        let mut packer = crate::Packer {
            buf: Vec::with_capacity($msl),
            sinc_buffer: [<$t>::EQUILIBRIUM; 128],
            tx: $tx.clone(),
        };

        let stream = if $config.sample_rate.0 == T::SAMPLE_RATE {
            #[allow(clippy::cast_possible_truncation)]
            $device.build_input_stream(
                &$config,
                move |data: &[$t], _info: &cpal::InputCallbackInfo| {
                    let data = data
                        .chunks_exact($config.channels as usize)
                        .map(|x| x.iter().sum::<$t>() / $config.channels as $t)
                        .map(dasp_sample::Sample::to_sample::<T::Data>);

                    packer.append(data);
                },
                move |err| {
                    error! {%err, "The mic error callback was called"};
                },
                None,
            )?
        } else {
            #[allow(clippy::cast_possible_truncation)]
            $device.build_input_stream(
                &$config,
                move |data: &[$t], _info: &cpal::InputCallbackInfo| {
                    let mono_data = data
                        .chunks_exact($config.channels as usize)
                        .map(|x| x.iter().sum::<$t>() / $config.channels as $t);

                    let data = dasp_signal::lift(mono_data, |signal| {
                        signal.from_hz_to_hz(
                            dasp_interpolate::sinc::Sinc::new(dasp_ring_buffer::Fixed::from(
                                packer.sinc_buffer,
                            )),
                            $config.sample_rate.0 as f64,
                            T::SAMPLE_RATE as f64,
                        )
                    })
                    .map(dasp_sample::Sample::to_sample::<T::Data>);

                    packer.append(data);
                },
                move |err| {
                    error! {%err, "The mic error callback was called"};
                },
                None,
            )?
        };
        stream.play()?;
        stream
    }};
}

pub(crate) use parse_data;

struct Packer<T, D> {
    buf: Vec<D>,
    sinc_buffer: [T; 128],
    tx: thingbuf::mpsc::blocking::Sender<Vec<D>, WithCapacity>,
}

impl<T, D> Packer<T, D> {
    pub fn append(&mut self, data: impl IntoIterator<Item = D>) {
        let mut data = data.into_iter().peekable();
        while data.peek().is_some() {
            let remaining_capacity = self.buf.capacity() - self.buf.len();
            if remaining_capacity == 0 {
                self.flush();
            } else {
                self.buf.extend(data.by_ref().take(remaining_capacity));
            };
        }
    }

    pub fn flush(&mut self) {
        match self.tx.try_send_ref() {
            Ok(mut send_ref) => {
                mem::swap(&mut *send_ref, &mut self.buf);
            }
            Err(err) => {
                warn!(%err, "Failed to send data to the Transcriber");
                self.buf.clear();
            }
        };
    }
}

impl<T, D> Drop for Packer<T, D> {
    fn drop(&mut self) {
        info!("Dropping the Packer");
        let _ = self.buf.pop();
        self.flush();
    }
}

#[derive(Debug, Error)]
pub enum StopError {
    #[error("No stream is currently running")]
    NoStreamRunning,
}

#[derive(Debug, Error)]
pub enum StartError {
    #[error("The transcriber is down, it may have paniced, call join() to see why it's down")]
    TranscriberDown,
    #[error("The transcriber is already running stop it before starting again")]
    TranscriberRunning,
    #[error("Failed to find an available input device")]
    DeviceError,
    #[error("Failed to find the selected device among the available devices")]
    SelectedDeviceNotFound,
    #[error("No (supported) config was found for the selected device")]
    NoConfigFound,
    #[error(transparent)]
    DeviceListError(#[from] cpal::DevicesError),
    #[error(transparent)]
    SupportedConfigListError(#[from] cpal::SupportedStreamConfigsError),
    #[error(transparent)]
    BuildStreamError(#[from] cpal::BuildStreamError),
    #[error(transparent)]
    PlayStreamError(#[from] cpal::PlayStreamError),
}

type MicStreamState = Arc<Mutex<Option<oneshot::Sender<()>>>>;

type TranscriberJoinHandle<T> = JoinHandle<Result<(), T>>;

type StartStream = (
    Settings,
    oneshot::Sender<Result<mpsc::Receiver<String>, StartError>>,
);

pub struct Transcriber<T>
where
    T: Model,
{
    stream_state: MicStreamState,
    ctrl_rx: mpsc::Receiver<StartStream>,
    common_model_params: CommonModelParams,
    model: T,
}

impl<T> Transcriber<T>
where
    T: Model,
{
    #[instrument(err(Display, level = Level::DEBUG))]
    pub fn blocking_new<D>(model_definition: D) -> Result<(Self, TranscriberHandle), D::Error>
    where
        D: ModelDefinition<Model = T> + Debug,
    {
        let stream = Arc::new(Mutex::new(None));

        let common_model_params = *model_definition.common_params();

        let (ctrl_tx, ctrl_rx) = mpsc::channel(1);

        let model: T = model_definition.blocking_try_to_model()?;

        Ok((
            Self {
                stream_state: Arc::clone(&stream),
                ctrl_rx,
                model,
                common_model_params,
            },
            TranscriberHandle {
                stream_state: stream,
                ctrl_tx,
            },
        ))
    }

    #[instrument(err(Display, level = Level::DEBUG))]
    pub async fn new<D>(model_definition: D) -> Result<(Self, TranscriberHandle), D::Error>
    where
        D: ModelDefinition<Model = T> + Debug,
    {
        let stream = Arc::new(Mutex::new(None));

        let common_model_params = *model_definition.common_params();

        let (ctrl_tx, ctrl_rx) = mpsc::channel(1);

        let model: T = model_definition.try_to_model().await?;

        Ok((
            Self {
                stream_state: Arc::clone(&stream),
                ctrl_rx,
                model,
                common_model_params,
            },
            TranscriberHandle {
                stream_state: stream,
                ctrl_tx,
            },
        ))
    }

    #[instrument(err(Display, level = Level::DEBUG))]
    pub fn blocking_spawn<D>(
        model_definition: D,
    ) -> Result<(TranscriberJoinHandle<T::Error>, TranscriberHandle), D::Error>
    where
        D: ModelDefinition<Model = T> + Debug,
    {
        let (transcriber, th) = Self::blocking_new(model_definition)?;
        let jh = thread::spawn(move || transcriber.run());
        Ok((jh, th))
    }

    #[instrument(err(Display, level = Level::DEBUG))]
    pub async fn spawn<D>(
        model_definition: D,
    ) -> Result<(TranscriberJoinHandle<T::Error>, TranscriberHandle), D::Error>
    where
        D: ModelDefinition<Model = T> + Debug,
    {
        let (transcriber, th) = Self::new(model_definition).await?;
        let jh = thread::spawn(move || transcriber.run());
        Ok((jh, th))
    }

    #[instrument(skip_all)]
    pub fn run(mut self) -> Result<(), T::Error> {
        while let Some((mic_settings, res_ch)) = self.ctrl_rx.blocking_recv() {
            let recycle = thingbuf::recycling::WithCapacity::new()
                .with_min_capacity(self.common_model_params.max_chunk_len())
                .with_max_capacity(self.common_model_params.max_chunk_len());
            let (data_tx, data_rx) = thingbuf::mpsc::blocking::with_recycle::<Vec<T::Data>, _>(
                self.common_model_params.data_buffer_size(),
                recycle,
            );
            let (string_tx, string_rx) =
                mpsc::channel(self.common_model_params.string_buffer_size());

            let (tmp_tx, tmp_rx) = oneshot::channel();

            let _jh = jod_thread::spawn(move || {
                match Self::create_stream(
                    &mic_settings,
                    &data_tx,
                    self.common_model_params.max_chunk_len(),
                ) {
                    Ok(_stream) => {
                        let (tx, rx) = oneshot::channel();
                        let _ = tmp_tx.send(Ok(tx));
                        let _ = rx.blocking_recv();
                    }
                    Err(err) => {
                        let _ = tmp_tx.send(Err(err));
                    }
                };
            });

            let create_stream = tmp_rx.blocking_recv().unwrap();

            match create_stream {
                Err(err) => {
                    if res_ch.send(Err(err)).is_err() {
                        warn!("Failed to send Stream creation failure response, receiver closed.");
                    };
                    break;
                }
                Ok(stream) => {
                    {
                        let mut guard = self.stream_state.lock().unwrap_or_else(|e| {
                                    error!("Ran into a poisoned Mutex when creating Stream, clearing the poison.");
                                    self.stream_state.clear_poison();
                                    e.into_inner()
                                });

                        if res_ch.send(Ok(string_rx)).is_ok() {
                            *guard = Some(stream);
                        } else {
                            warn!(
                                "Failed to send Stream creation success response, receiver closed."
                            );
                            break;
                        };
                    };

                    while let Ok((_, res_ch)) = self.ctrl_rx.try_recv() {
                        if res_ch.send(Err(StartError::TranscriberRunning)).is_err() {
                            warn!(
                                "Failed to send Stream creation failure response, receiver closed."
                            );
                        };
                    }

                    while let Some(mut data) = data_rx.recv_ref() {
                        let final_chunk = data.capacity() > data.len();
                        let string = match self.model.transcribe(&mut *data, final_chunk) {
                            Ok(string) => string,
                            Err(err) => {
                                error!(%err, "The Transcriber ran into an unrecoverable error.");
                                {
                                    let mut guard =  self.stream_state.lock().unwrap_or_else(|e| {
                                        error!("Ran into a poisoned Mutex when dropping the Stream on transcriber error, clearing the poison.");
                                        self.stream_state.clear_poison();
                                        e.into_inner()
                                    });
                                    *guard = None;
                                };
                                return Err(err);
                            }
                        };
                        if string_tx.blocking_send(string).is_err() {
                            {
                                let mut guard =  self.stream_state.lock().unwrap_or_else(|e| {
                                        error!("Ran into a poisoned Mutex when dropping the Stream on closed Reciever, clearing the poison.");
                                        self.stream_state.clear_poison();
                                        e.into_inner()
                                    });
                                *guard = None;
                            };
                            break;
                        };
                    }
                }
            }
        }
        Ok(())
    }
}

impl<T> Transcriber<T>
where
    T: Model,
{
    #[instrument(level = Level::DEBUG, skip(data_tx), err(Display, level = Level::TRACE))]
    fn create_stream(
        mic_settings: &Settings,
        data_tx: &thingbuf::mpsc::blocking::Sender<Vec<T::Data>, WithCapacity>,
        max_chunk_len: usize,
    ) -> Result<cpal::Stream, StartError> {
        let host = cpal::default_host();

        let device = match mic_settings.selected_device {
            Some(ref selected_device) => match host.input_devices()?.find(|device| {
                device
                    .name()
                    .map(|device_name| device_name == *selected_device)
                    .unwrap_or(false)
            }) {
                Some(x) => Some(x),
                None => match mic_settings.on_error {
                    crate::mic::OnError::Error => return Err(StartError::SelectedDeviceNotFound),
                    crate::mic::OnError::TryDefault => host.default_input_device(),
                },
            },
            None => host.default_input_device(),
        }
        .ok_or(StartError::DeviceError)?;

        let mut input_conf = device
            .supported_input_configs()?
            .collect::<Vec<SupportedStreamConfigRange>>();
        input_conf.sort_by(|lhs, rhs| Self::cmp_mic_config(lhs, rhs));

        loop {
            let Some(config) = input_conf.pop() else {
                break Err(StartError::NoConfigFound);
            };

            let sample_format = config.sample_format();
            let config = config
                .try_with_sample_rate(SampleRate(T::SAMPLE_RATE))
                .unwrap_or_else(|| config.with_max_sample_rate())
                .config();

            break Ok(match sample_format {
                cpal::SampleFormat::I8 => parse_data!(i8, device, config, data_tx, max_chunk_len),
                cpal::SampleFormat::I16 => parse_data!(i16, device, config, data_tx, max_chunk_len),
                cpal::SampleFormat::I32 => parse_data!(i32, device, config, data_tx, max_chunk_len),
                cpal::SampleFormat::I64 => parse_data!(i64, device, config, data_tx, max_chunk_len),
                cpal::SampleFormat::U8 => parse_data!(u8, device, config, data_tx, max_chunk_len),
                cpal::SampleFormat::U16 => parse_data!(u16, device, config, data_tx, max_chunk_len),
                cpal::SampleFormat::U32 => parse_data!(u32, device, config, data_tx, max_chunk_len),
                cpal::SampleFormat::U64 => parse_data!(u64, device, config, data_tx, max_chunk_len),
                cpal::SampleFormat::F32 => parse_data!(f32, device, config, data_tx, max_chunk_len),
                cpal::SampleFormat::F64 => parse_data!(f64, device, config, data_tx, max_chunk_len),
                _ => continue,
            });
        }
    }

    #[instrument(level = Level::TRACE, ret)]
    fn cmp_mic_config(
        lhs: &SupportedStreamConfigRange,
        rhs: &SupportedStreamConfigRange,
    ) -> Ordering {
        let lhs_sample_rate = lhs.max_sample_rate() >= SampleRate(T::SAMPLE_RATE)
            && SampleRate(T::SAMPLE_RATE) >= lhs.min_sample_rate();

        let rhs_sample_rate = rhs.max_sample_rate() >= SampleRate(T::SAMPLE_RATE)
            && SampleRate(T::SAMPLE_RATE) >= rhs.min_sample_rate();

        if lhs_sample_rate && rhs_sample_rate {
            let cmp_format = (lhs.sample_format() == (T::Data::to_sample_fromat()))
                .cmp(&(rhs.sample_format() == (T::Data::to_sample_fromat())));
            if cmp_format != Equal {
                return cmp_format;
            };
        } else {
            let cmp_sample_rate = lhs_sample_rate.cmp(&rhs_sample_rate);
            if cmp_sample_rate != Equal {
                return cmp_sample_rate;
            };

            let cmp_format_f64 = (lhs.sample_format() == cpal::SampleFormat::F64)
                .cmp(&(rhs.sample_format() == cpal::SampleFormat::F64));
            if cmp_format_f64 != Equal {
                return cmp_format_f64;
            };

            let cmp_float = (lhs.sample_format().is_float()).cmp(&rhs.sample_format().is_float());
            if cmp_float != Equal {
                return cmp_float;
            }
        }

        let cmp_mono = (lhs.channels() == 1).cmp(&(rhs.channels() == 1));
        if cmp_mono != Equal {
            return cmp_mono;
        };

        Equal
    }
}

#[must_use = "The transcriber will terminate if this is droped"]
#[derive(Debug, Clone)]
pub struct TranscriberHandle {
    stream_state: MicStreamState,
    ctrl_tx: mpsc::Sender<StartStream>,
}

impl TranscriberHandle {
    #[instrument(skip(self), err(Display, level = Level::DEBUG))]
    pub async fn start(
        &self,
        mic_settings: Settings,
    ) -> Result<mpsc::Receiver<String>, StartError> {
        let is_down = self
            .stream_state
            .lock()
            .unwrap_or_else(|e| {
                error!(
                "Ran into a poisoned Mutex when attempting to start a Stream, clearing the poison."
            );
                self.stream_state.clear_poison();
                let mut guard = e.into_inner();
                *guard = None;
                guard
            })
            .is_none();

        if is_down {
            let (res_tx, res_rx) = oneshot::channel();

            self.ctrl_tx
                .send((mic_settings, res_tx))
                .await
                .map_err(|_| StartError::TranscriberDown)?;

            Ok(res_rx.await.map_err(|_| StartError::TranscriberDown)??)
        } else {
            Err(StartError::TranscriberRunning)
        }
    }

    #[instrument(skip(self), err(Display, level = Level::DEBUG))]
    pub fn blocking_start(
        &self,
        mic_settings: Settings,
    ) -> Result<mpsc::Receiver<String>, StartError> {
        let is_down = self
            .stream_state
            .lock()
            .unwrap_or_else(|e| {
                error!(
                "Ran into a poisoned Mutex when attempting to start a Stream, clearing the poison."
            );
                self.stream_state.clear_poison();
                let mut guard = e.into_inner();
                *guard = None;
                guard
            })
            .is_none();

        if is_down {
            let (res_tx, res_rx) = oneshot::channel();

            self.ctrl_tx
                .blocking_send((mic_settings, res_tx))
                .map_err(|_| StartError::TranscriberDown)?;

            Ok(res_rx
                .blocking_recv()
                .map_err(|_| StartError::TranscriberDown)??)
        } else {
            Err(StartError::TranscriberRunning)
        }
    }

    #[instrument(skip(self), err(Display, level = Level::DEBUG))]
    pub fn stop(&self) -> Result<(), StopError> {
        match self.stream_state.lock() {
            Ok(mut guard) if guard.is_some() => {
                *guard = None;
                Ok(())
            }
            Ok(_) => Err(StopError::NoStreamRunning),
            Err(err) => {
                warn!("Ran into a poisoned Mutex when dropping the Stream from a TranscriberHandle, clearing the poison.");
                self.stream_state.clear_poison();
                let mut guard = err.into_inner();
                *guard = None;
                Ok(())
            }
        }
    }
}
