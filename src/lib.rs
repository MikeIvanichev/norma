pub mod dtype;
pub mod mic;
pub mod model;

use std::{
    cmp::Ordering::{self, Equal},
    mem,
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
};

use crate::dtype::DType;

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SampleRate, SupportedStreamConfigRange,
};
use mic::MicSettings;
use model::{CommonModelParams, Model, ModelDefinition};
use thingbuf::recycling::WithCapacity;
use thiserror::Error;
use tracing::{error, warn};

use tokio::sync::{mpsc, oneshot};

macro_rules! parse_data {
    ($t:ty, $device:ident, $config:ident, $tx: ident, $msl: ident) => {{
        use dasp_frame::Frame;
        use dasp_signal::Signal;
        use tracing::error;

        let mut packer = crate::Packer {
            buff: Vec::with_capacity($msl),
            sinc_buffer: [<$t>::EQUILIBRIUM; 128],
            tx: $tx.clone(),
        };

        let stream = $device.build_input_stream(
            &$config,
            move |data: &[$t], _info: &cpal::InputCallbackInfo| {
                let mono_data = data
                    .chunks_exact($config.channels as usize)
                    .map(|x| x.iter().sum::<$t>() / $config.channels as $t);

                //TODO Look into optimizing out calling the sample rate conversion of each call and
                //the associated allocations
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
                error! {"The mic error callback was called with: {:?}", err};
            },
            None,
        )?;
        stream.play()?;
        stream
    }};
}

pub(crate) use parse_data;

struct Packer<T, D> {
    buff: Vec<D>,
    sinc_buffer: [T; 128],
    tx: thingbuf::mpsc::blocking::Sender<Vec<D>, WithCapacity>,
}

impl<T, D> Packer<T, D> {
    pub fn append(&mut self, data: impl IntoIterator<Item = D>) {
        let mut data = data.into_iter().peekable();
        while data.peek().is_some() {
            let remaining_capacity = self.buff.capacity() - self.buff.len();
            if remaining_capacity == 0 {
                self.flush();
            } else {
                self.buff.extend(data.by_ref().take(remaining_capacity));
            };
        }
    }

    pub fn flush(&mut self) {
        match self.tx.try_send_ref() {
            Ok(mut send_ref) => {
                mem::swap(&mut *send_ref, &mut self.buff);
            }
            Err(err) => {
                warn!("Failed to send data to the Transcriber, {err}.");
                self.buff.clear();
            }
        };
    }
}

impl<T, D> Drop for Packer<T, D> {
    fn drop(&mut self) {
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

type MicStreamState = Arc<Mutex<Option<cpal::Stream>>>;

type StartStream = (
    MicSettings,
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

unsafe impl<T: Model> Send for Transcriber<T> {}

impl<T> Transcriber<T>
where
    T: Model,
{
    pub fn new<D>(model_definition: D) -> Result<(Self, TranscriberHandle), D::Error>
    where
        D: ModelDefinition<Model = T>,
    {
        let stream = Arc::new(Mutex::new(None));

        let common_model_params = model_definition.common_params();

        let (ctrl_tx, ctrl_rx) = mpsc::channel(1);

        let model: T = model_definition.try_into()?;

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

    pub fn spawn<D>(model_definition: D) -> Result<(JoinHandle<()>, TranscriberHandle), D::Error>
    where
        D: ModelDefinition<Model = T>,
    {
        let (transcriber, th) = Self::new(model_definition)?;
        let jh = thread::spawn(move || transcriber.run());
        Ok((jh, th))
    }

    pub fn run(mut self) {
        while let Some((mic_settings, res_ch)) = self.ctrl_rx.blocking_recv() {
            let recycle = thingbuf::recycling::WithCapacity::new()
                .with_min_capacity(self.common_model_params.max_sample_len)
                .with_max_capacity(self.common_model_params.max_sample_len);
            let (data_tx, data_rx) = thingbuf::mpsc::blocking::with_recycle::<Vec<T::Data>, _>(
                self.common_model_params.data_buffer,
                recycle,
            );
            let (string_tx, string_rx) = mpsc::channel(self.common_model_params.string_buffer);

            match self.create_stream(mic_settings, data_tx) {
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
                        let string = self.model.transcribe(&mut *data);
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
            self.model.clear_context();
        }
    }
}

impl<T> Transcriber<T>
where
    T: Model,
{
    fn create_stream(
        &self,
        mic_settings: MicSettings,
        data_tx: thingbuf::mpsc::blocking::Sender<Vec<T::Data>, WithCapacity>,
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

            let msl = self.common_model_params.max_sample_len;

            break Ok(match sample_format {
                cpal::SampleFormat::I8 => parse_data!(i8, device, config, data_tx, msl),
                cpal::SampleFormat::I16 => parse_data!(i16, device, config, data_tx, msl),
                cpal::SampleFormat::I32 => parse_data!(i32, device, config, data_tx, msl),
                cpal::SampleFormat::I64 => parse_data!(i64, device, config, data_tx, msl),
                cpal::SampleFormat::U8 => parse_data!(u8, device, config, data_tx, msl),
                cpal::SampleFormat::U16 => parse_data!(u16, device, config, data_tx, msl),
                cpal::SampleFormat::U32 => parse_data!(u32, device, config, data_tx, msl),
                cpal::SampleFormat::U64 => parse_data!(u64, device, config, data_tx, msl),
                cpal::SampleFormat::F32 => parse_data!(f32, device, config, data_tx, msl),
                cpal::SampleFormat::F64 => parse_data!(f64, device, config, data_tx, msl),
                _ => continue,
            });
        }
    }

    fn cmp_mic_config(
        lhs: &SupportedStreamConfigRange,
        rhs: &SupportedStreamConfigRange,
    ) -> Ordering {
        let cmp_sample_rate = (lhs.max_sample_rate() >= SampleRate(T::SAMPLE_RATE)
            && SampleRate(T::SAMPLE_RATE) >= lhs.min_sample_rate())
        .cmp(
            &(rhs.max_sample_rate() >= SampleRate(T::SAMPLE_RATE)
                && SampleRate(T::SAMPLE_RATE) >= rhs.min_sample_rate()),
        );
        if cmp_sample_rate != Equal {
            return cmp_sample_rate;
        };

        let cmp_format = (lhs.sample_format() == (T::Data::to_sample_fromat()))
            .cmp(&(rhs.sample_format() == (T::Data::to_sample_fromat())));
        if cmp_format != Equal {
            return cmp_format;
        };

        // If we got this far we already know taht we have to interpolate
        // Sinc will convert any incoming dtype to f64, so we might as well record in f64
        let cmp_format_f64 = (lhs.sample_format() == cpal::SampleFormat::F64)
            .cmp(&(rhs.sample_format() == cpal::SampleFormat::F64));
        if cmp_format_f64 != Equal {
            return cmp_format_f64;
        };

        let cmp_mono = (lhs.channels() == 1).cmp(&(rhs.channels() == 1));
        if cmp_mono != Equal {
            return cmp_mono;
        };

        Equal
    }
}

pub struct TranscriberHandle {
    stream_state: MicStreamState,
    ctrl_tx: mpsc::Sender<StartStream>,
}

unsafe impl Send for TranscriberHandle {}

impl TranscriberHandle {
    pub async fn start(
        &self,
        mic_settings: MicSettings,
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

    pub fn start_blocking(
        &self,
        mic_settings: MicSettings,
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
