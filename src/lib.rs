pub mod dtype;
pub mod mic;
pub mod model;

use std::{
    cmp::Ordering,
    io::Read,
    ops::{Deref, DerefMut},
    sync::mpsc::{channel, Receiver, Sender},
    thread::{self, JoinHandle},
    time::Duration,
};

use crate::dtype::DType;

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    InputCallbackInfo, SampleRate, Stream, SupportedStreamConfigRange,
};
use dasp_frame::Frame;
use dasp_ring_buffer::SliceMut;
use dasp_signal::Signal;
use mic::MicSettings;
use model::{ModelDefinition, ModelInput};
use serde::{Deserialize, Serialize};
use thiserror::Error;

macro_rules! parse_data {
    ($t:ty, $device:ident, $config:ident, $tx: ident) => {{
        let mut packer = Packer {
            rb: Vec::with_capacity(T::MAX_SAMPLE_LEN + 1_000),
            sinc_buffer: [<$t>::EQUILIBRIUM; 128],
            tx: $tx.clone(),
        };
        $device.build_input_stream(
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

                packer.rb.extend(data);
                if packer.rb.len() >= T::MAX_SAMPLE_LEN {
                    packer.flush();
                }
            },
            move |err| $tx.send(InputMsg::StreamError(err)).unwrap(),
            None,
        )?
    }};
}

struct Packer<T, D> {
    rb: Vec<D>,
    sinc_buffer: [T; 128],
    tx: Sender<InputMsg<D>>,
}

impl<T, D> Packer<T, D> {
    pub fn flush(&mut self) {
        let data = Box::from_iter(self.rb.drain(..));
        self.tx.send(InputMsg::Data(data)).expect("The receiver for the input channel has disconnected before the sender. This should be impossible!");
    }
}

impl<T, D> Drop for Packer<T, D> {
    fn drop(&mut self) {
        self.flush();
        self.tx.send(InputMsg::StreamEnd).expect("The receiver for the input channel has disconnected before the sender. This should be impossible!");
    }
}

enum InputMsg<T> {
    Start,
    StreamEnd,
    StreamError(cpal::StreamError),
    Data(Box<[T]>),
}

enum OutputMsg {
    End,
    TranscriptionError(TranscriptionError),
    Data(String),
}

enum State<T> {
    Stoped,
    Idle {
        in_tx: Sender<InputMsg<T>>,
        out_rx: Receiver<OutputMsg>,
        jh: JoinHandle<()>,
    },
    Running {
        stream: Stream,
        in_tx: Sender<InputMsg<T>>,
        out_rx: Receiver<OutputMsg>,
        jh: JoinHandle<()>,
    },
}

pub struct Transcriber<T>
where
    T: ModelDefinition,
{
    model_definition: T,
    state: State<T::Data>,
}

impl<T> Transcriber<T>
where
    T: ModelDefinition,
{
    pub fn new(model_definition: T) -> Self {
        todo!()
    }

    pub fn transcribe(&mut self, mic_settings: MicSettings) -> Result<(), StartError> {
        match &mut self.state {
            State::Stoped => {
                let (in_tx, in_rx) = channel();
                let (out_tx, out_rx) = channel();
                let mut model = self
                    .model_definition
                    .run()
                    .map_err(|err| StartError::FailedToStartModel(err.to_string()))?;
                let stream = Self::create_stream(mic_settings, in_tx.clone())?;
                stream.play()?;
                let jh = thread::spawn(move || {
                    const SEND_MSG_FAIL: &str = "The receiver for the output channel has disconnected before the sender. This should be impossible!";
                    let mut stream_error = None;
                    let mut model_error = false;

                    loop {
                        if let Ok(val) = in_rx.recv() {
                            if model_error {
                                continue;
                            }

                            match val {
                                InputMsg::Data(mut data) => {
                                    match model(ModelInput::Data(data.slice_mut())) {
                                        Ok(res) => {
                                            out_tx.send(OutputMsg::Data(res)).expect(SEND_MSG_FAIL);
                                        }
                                        Err(err) => {
                                            model_error = true;
                                            out_tx
                                                .send(OutputMsg::TranscriptionError(
                                                    TranscriptionError::ModelError(err.to_string()),
                                                ))
                                                .expect(SEND_MSG_FAIL);
                                        }
                                    };
                                }
                                InputMsg::StreamEnd => {
                                    let msg = stream_error
                                        .take()
                                        .map(OutputMsg::TranscriptionError)
                                        .unwrap_or(OutputMsg::End);
                                    out_tx.send(msg).expect(SEND_MSG_FAIL);
                                }
                                InputMsg::StreamError(err) => {
                                    stream_error = Some(TranscriptionError::StreamError(err))
                                }
                                InputMsg::Start => {
                                    if let Err(err) = model(ModelInput::ClearContext) {
                                        model_error = true;
                                        out_tx
                                            .send(OutputMsg::TranscriptionError(
                                                TranscriptionError::ModelError(err.to_string()),
                                            ))
                                            .expect(SEND_MSG_FAIL);
                                    }
                                }
                            }
                        } else {
                            return;
                        }
                    }
                });
                self.state = State::Running {
                    stream,
                    in_tx,
                    out_rx,
                    jh,
                };
                Ok(())
            }
            State::Idle {
                in_tx: comm_tx,
                out_rx: string_rx,
                jh,
            } => todo!(),
            State::Running { .. } => Ok(()),
        }
    }

    pub fn stop(&mut self) {
        todo!()
    }

    pub fn idle(&mut self) {
        todo!()
    }

    pub fn recv(&self) -> String {
        todo!();
    }

    fn cmp_config(lhs: &SupportedStreamConfigRange, rhs: &SupportedStreamConfigRange) -> Ordering {
        use std::cmp::Ordering::Equal;

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

        let cmp_mono = (lhs.channels() == 1).cmp(&(rhs.channels() == 1));
        if cmp_mono != Equal {
            return cmp_mono;
        };

        Equal
    }

    //pub fn start_transcription_old(&mut self) -> Result<(), StartError> {
    //    if let Some(ref handle) = self.actor {
    //        //TODO add a check to see if the actor is still alive
    //        let stream = self.create_stream(input_conf, &device, handle.transport_tx.clone())?;
    //        stream.play()?;

    //        self.input_stream = Some(stream);
    //    } else {
    //        let (transport_tx, transport_rx) = channel();
    //        let stream = self.create_stream(input_conf, &device, transport_tx.clone())?;
    //        stream.play()?;
    //        let (string_tx, string_rx) = channel();
    //        let jh = thread::spawn(
    //            self.model_definition
    //                .run(transport_rx, string_tx)
    //                .map_err(|e| StartError::FailedToStartModel(e.to_string()))?,
    //        );

    //        self.actor = Some(Actor {
    //            jh,
    //            transport_tx,
    //            string_rx,
    //        });
    //        self.input_stream = Some(stream);
    //    };

    //    Ok(())
    //}

    fn create_stream(
        mic_settings: MicSettings,
        transport_tx: Sender<InputMsg<T::Data>>,
    ) -> Result<Stream, StartError> {
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
                    mic::OnError::Error => return Err(StartError::SelectedDeviceNotFound),
                    mic::OnError::TryDefault => host.default_input_device(),
                },
            },
            None => host.default_input_device(),
        }
        .ok_or(StartError::DeviceError)?;

        let mut input_conf = device
            .supported_input_configs()?
            .collect::<Vec<SupportedStreamConfigRange>>();
        input_conf.sort_by(|lhs, rhs| Self::cmp_config(lhs, rhs));

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
                cpal::SampleFormat::I8 => parse_data!(i8, device, config, transport_tx),
                cpal::SampleFormat::I16 => parse_data!(i16, device, config, transport_tx),
                cpal::SampleFormat::I32 => parse_data!(i32, device, config, transport_tx),
                cpal::SampleFormat::I64 => parse_data!(i64, device, config, transport_tx),
                cpal::SampleFormat::U8 => parse_data!(u8, device, config, transport_tx),
                cpal::SampleFormat::U16 => parse_data!(u16, device, config, transport_tx),
                cpal::SampleFormat::U32 => parse_data!(u32, device, config, transport_tx),
                cpal::SampleFormat::U64 => parse_data!(u64, device, config, transport_tx),
                cpal::SampleFormat::F32 => parse_data!(f32, device, config, transport_tx),
                cpal::SampleFormat::F64 => parse_data!(f64, device, config, transport_tx),
                _ => continue,
            });
        }
    }

    //pub fn end_transcription(&mut self) -> Result<(), EndError> {
    //    let Some(ref handle) = self.actor else {
    //        return Err(EndError::NoRunningModel);
    //    };
    //    let Some(ref stream) = self.input_stream else {
    //        return Err(EndError::NoActiveStream);
    //    };

    //    match self.mode {
    //        Mode::Oneshot => todo!(),
    //        Mode::Presist(_) => {
    //            todo!()
    //        }
    //    }
    //}

    //pub fn set_mode(&mut self, mode: Mode) {
    //    if mode == self.mode {
    //        return;
    //    }

    //    if let Mode::Presist(_) = mode {
    //        self.mode = mode;
    //    } else {
    //        // We should kill all running models.
    //        todo!()
    //    }
    //}
}

impl<T> From<T> for Transcriber<T>
where
    T: ModelDefinition,
{
    fn from(value: T) -> Self {
        todo!()
    }
}

#[derive(Debug, Error)]
pub enum StartError {
    #[error("Failed to find an available input device")]
    DeviceError,
    #[error("Failed to find the selected device among the available devices")]
    SelectedDeviceNotFound,
    #[error("No (supported) config was found for the selected device")]
    NoConfigFound,
    #[error("Failed to start the model becouse: {0}")]
    FailedToStartModel(String),
    #[error(transparent)]
    DeviceListError(#[from] cpal::DevicesError),
    #[error(transparent)]
    SupportedConfigListError(#[from] cpal::SupportedStreamConfigsError),
    #[error(transparent)]
    BuildStreamError(#[from] cpal::BuildStreamError),
    #[error(transparent)]
    PlayStreamError(#[from] cpal::PlayStreamError),
}

#[derive(Debug, Error)]
pub enum TranscriptionError {
    #[error(transparent)]
    StreamError(#[from] cpal::StreamError),
    #[error("The model has crashed becouse: {0}")]
    ModelError(String),
}

#[derive(Debug, Error)]
pub enum ReceiveError {
    #[error("The transcribe is not currently running, call transcribe to start it")]
    TranscriberNotRunning,
    #[error("The stream is not currently running, call transcribe to start it")]
    TranscriberIdle,
    #[error(transparent)]
    TranscriptionError(#[from] TranscriptionError),
}

mod private {
    pub trait Sealed {}
}
