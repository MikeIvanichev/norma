pub mod dtype;
pub mod mic;
pub mod model;

use std::{
    cmp::Ordering,
    sync::mpsc::{channel, Receiver, Sender},
    thread::{self, JoinHandle},
    time::Duration,
};

use crate::dtype::DType;

use candle_core::Tensor;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SampleRate, Stream, SupportedStreamConfigRange,
};
use dasp_frame::Frame;
use dasp_signal::Signal;
use mic::MicSettings;
use model::ModelDefinition;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub struct Handle {
    jh: JoinHandle<()>,
    tensor_tx: Sender<Tensor>,
    string_rx: Receiver<String>,
}

pub struct Norma<T>
where
    T: ModelDefinition,
{
    model_definition: T,
    mic_settings: MicSettings,
    mode: Mode,
    _stream: Option<Stream>,
    _handle: Option<Handle>,
}

macro_rules! parse_data {
    ($t:ty, $device:ident, $config:ident, $packer: expr) => {{
        let mut packer = $packer;
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
                            [<$t>::EQUILIBRIUM; 128],
                        )),
                        $config.sample_rate.0 as f64,
                        T::SAMPLE_RATE as f64,
                    )
                })
                .map(dasp_sample::Sample::to_sample::<T::DType>)
                //TODO Look into optimizing out this call to collect Vec
                .collect::<Vec<T::DType>>();

                packer(data);
            },
            |err| panic!("{err}"),
            None,
        )?
    }};
}

impl<T> Norma<T>
where
    T: ModelDefinition,
{
    pub fn new() -> Self {
        todo!()
    }

    fn cmp_config(
        &self,
        lhs: &SupportedStreamConfigRange,
        rhs: &SupportedStreamConfigRange,
    ) -> Ordering {
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

        let cmp_format = (lhs.sample_format() == (T::DType::to_sample_fromat()))
            .cmp(&(rhs.sample_format() == (T::DType::to_sample_fromat())));
        if cmp_format != Equal {
            return cmp_format;
        };

        let cmp_mono = (lhs.channels() == 1).cmp(&(rhs.channels() == 1));
        if cmp_mono != Equal {
            return cmp_mono;
        };

        Equal
    }

    pub fn start_transcription(&mut self) -> Result<(), StartError> {
        if self._stream.is_some() {
            return Err(StartError::TranscriptionRunning);
        }

        let host = cpal::default_host();

        let device = match self.mic_settings.selected_device {
            Some(ref selected_device) => match host.input_devices()?.find(|device| {
                device
                    .name()
                    .map(|device_name| device_name == *selected_device)
                    .unwrap_or(false)
            }) {
                Some(x) => Some(x),
                None => match self.mic_settings.on_error {
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

        input_conf.sort_by(|lhs, rhs| self.cmp_config(lhs, rhs));

        if let Some(ref handle) = self._handle {
            let stream = self.create_stream(input_conf, device, handle.tensor_tx.clone())?;
            stream.play()?;
        } else {
            let (tensor_tx, tensor_rx) = channel();
            let stream = self.create_stream(input_conf, device, tensor_tx.clone())?;
            stream.play()?;
            let (string_tx, string_rx) = channel();
            let jh = thread::spawn(self.model_definition.run(tensor_rx, string_tx));
            self._handle = Some(Handle {
                jh,
                tensor_tx,
                string_rx,
            });
        };

        Ok(())
    }

    fn create_stream(
        &self,
        mut input_conf: Vec<SupportedStreamConfigRange>,
        device: cpal::Device,
        tensor_tx: Sender<Tensor>,
    ) -> Result<Stream, StartError> {
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
                cpal::SampleFormat::I8 => parse_data!(
                    i8,
                    device,
                    config,
                    self.model_definition.pack_data(tensor_tx)
                ),
                cpal::SampleFormat::I16 => parse_data!(
                    i16,
                    device,
                    config,
                    self.model_definition.pack_data(tensor_tx)
                ),
                cpal::SampleFormat::I32 => parse_data!(
                    i32,
                    device,
                    config,
                    self.model_definition.pack_data(tensor_tx)
                ),
                cpal::SampleFormat::I64 => parse_data!(
                    i64,
                    device,
                    config,
                    self.model_definition.pack_data(tensor_tx)
                ),
                cpal::SampleFormat::U8 => parse_data!(
                    u8,
                    device,
                    config,
                    self.model_definition.pack_data(tensor_tx)
                ),
                cpal::SampleFormat::U16 => parse_data!(
                    u16,
                    device,
                    config,
                    self.model_definition.pack_data(tensor_tx)
                ),
                cpal::SampleFormat::U32 => parse_data!(
                    u32,
                    device,
                    config,
                    self.model_definition.pack_data(tensor_tx)
                ),
                cpal::SampleFormat::U64 => parse_data!(
                    u64,
                    device,
                    config,
                    self.model_definition.pack_data(tensor_tx)
                ),
                cpal::SampleFormat::F32 => parse_data!(
                    f32,
                    device,
                    config,
                    self.model_definition.pack_data(tensor_tx)
                ),
                cpal::SampleFormat::F64 => parse_data!(
                    f64,
                    device,
                    config,
                    self.model_definition.pack_data(tensor_tx)
                ),
                _ => continue,
            });
        }
    }

    pub fn end_transcription(&mut self) -> Result<(), EndError> {
        let Some(ref handle) = self._handle else {
            return Err(EndError::NoRunningModel);
        };
        let Some(ref stream) = self._stream else {
            return Err(EndError::NoActiveStream);
        };

        match self.mode {
            Mode::Oneshot => todo!(),
            Mode::Presist(_) => {
                todo!()
            }
        }
    }

    pub fn set_mode(&mut self, mode: Mode) {
        if mode == self.mode {
            return;
        }

        if let Mode::Presist(_) = mode {
            self.mode = mode
        } else {
            // We should kill all running models.
            todo!()
        }
    }
}

#[derive(Debug, Error)]
pub enum StartError {
    #[error("The transcription is already running")]
    TranscriptionRunning,

    #[error("Failed to find an available input device")]
    DeviceError,

    #[error("Failed to find the selected device among the available devices")]
    SelectedDeviceNotFound,

    #[error("No (supported) config was found for the selected device")]
    NoConfigFound,

    #[error("Failed to list available devices")]
    DeviceListError(#[from] cpal::DevicesError),

    #[error("Failed to list available configs for the selected input device")]
    SupportedConfigListError(#[from] cpal::SupportedStreamConfigsError),

    #[error("An error ocured when building the stream")]
    BuildStreamError(#[from] cpal::BuildStreamError),

    #[error("Failed to play stream (explicitly start the recording)")]
    PlayStreamError(#[from] cpal::PlayStreamError),
}

pub enum EndError {
    NoRunningModel,
    NoActiveStream,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Mode {
    #[default]
    Oneshot,
    Presist(Duration),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Device {
    #[default]
    Cpu,
    Cuda(usize),
    Metal(usize),
}

mod private {
    pub trait Sealed {}
}
