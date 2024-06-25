use cpal::{
    default_host,
    traits::{DeviceTrait, HostTrait},
    SampleRate, SupportedStreamConfigRange,
};

use crate::{
    mic::MicSettings,
    model::{CommonModelParams, Model, ModelDefinition},
    DType, StartError, Stream,
};

use super::parse_data;

use std::{
    cmp::Ordering::{self, Equal},
    sync::{
        mpsc::{channel, Receiver, Sender},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
};

type MicStream = Arc<Mutex<Option<cpal::Stream>>>;

enum CtrlMsg {
    StartStream {
        mic_settings: MicSettings,
        res_ch: Sender<Result<Receiver<String>, StartError>>,
    },
}

struct DataChan<T> {
    data_rx: thingbuf::mpsc::blocking::Receiver<Vec<T>>,
    error_rx: Receiver<Option<cpal::StreamError>>,
}

pub struct Transcriber<T>
where
    T: Model,
{
    stream: MicStream,
    ctrl_rx: Receiver<CtrlMsg>,
    data_channel: Option<DataChan<T::Data>>,
    model: T,
    common_model_params: CommonModelParams,
}

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

        let (ctrl_tx, ctrl_rx) = channel();

        let model: T = model_definition.try_into()?;

        Ok((
            Self {
                stream: Arc::clone(&stream),
                ctrl_rx,
                model,
                common_model_params,
                data_channel: None,
            },
            TranscriberHandle { stream, ctrl_tx },
        ))
    }

    pub fn spawn<D>(model_definition: D) -> (JoinHandle<()>, TranscriberHandle)
    where
        D: ModelDefinition<Model = T>,
    {
        todo!()
    }

    pub async fn run(mut self) -> () {
        while let Ok(command) = self.ctrl_rx.recv() {
            match command {
                CtrlMsg::StartStream {
                    mic_settings,
                    res_ch,
                } => {
                    // try to create a new stream ->
                    // return res using chan, if chan is closed dump Stream ->

                    // Start data loop ->

                    self.model.clear_context();
                }
            }
        }
    }
}

impl<T> Transcriber<T>
where
    T: Model,
{
    fn transcribe(&mut self) -> String {
        todo!()
    }

    fn create_stream(&self, mic_settings: MicSettings) -> Result<cpal::Stream, StartError> {
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

        let recycle = thingbuf::recycling::WithCapacity::new()
            .with_min_capacity(self.common_model_params.max_sample_len)
            .with_max_capacity(self.common_model_params.max_sample_len);
        let (data_tx, data_rx) = thingbuf::mpsc::blocking::with_recycle::<Vec<T::Data>, _>(
            self.common_model_params.data_buffer,
            recycle,
        );

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
    stream: MicStream,
    ctrl_tx: Sender<CtrlMsg>,
}
