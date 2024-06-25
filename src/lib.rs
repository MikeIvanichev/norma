pub mod blocking;
pub mod dtype;
pub mod mic;
pub mod model;

use std::{
    cmp::Ordering,
    collections::HashMap,
    marker::PhantomData,
    mem,
    ops::DerefMut,
    sync::{mpsc::channel, Arc},
    thread::{self, JoinHandle},
};

use crate::dtype::DType;

use candle_core::WithDType;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SampleRate, Stream, SupportedStreamConfigRange,
};
use mic::MicSettings;
use model::{dummy::DummyDef, CommonModelParams, Model, ModelDefinition, ModelInput};
use thingbuf::{mpsc::blocking::SendRef, recycling::WithCapacity};
use thiserror::Error;

macro_rules! parse_data {
    ($t:ty, $device:ident, $config:ident, $tx: ident, $msl: ident) => {{
        use dasp_frame::Frame;
        use dasp_ring_buffer::SliceMut;
        use dasp_signal::Signal;

        let mut packer = crate::Packer {
            slot: $tx
                .send_ref()
                .expect("This is the first send, this should always succeed"),
            threshold: $msl,
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

                // packer.rb.extend(data);
                // if packer.rb.len() >= packer.threshold {
                //     packer.flush();
                // }
            },
            move |err| {
                todo!();
                //let _ = $tx.send(Msg::Error(err));
            },
            None,
        )?
    }};
}

pub(crate) use parse_data;

struct Packer<'a, T, D> {
    slot: SendRef<'a, Vec<D>>,
    threshold: usize,
    sinc_buffer: [T; 128],
    tx: thingbuf::mpsc::blocking::Sender<Vec<D>, WithCapacity>,
}

impl<'a, T, D> Packer<'a, T, D> {
    pub fn flush(self) {
        todo!();
        //let data = Box::from_iter(self.rb.drain(..self.threshold.min(self.rb.len())));
        //let _ = self.tx.send(DataMsg::Data(data));
    }
}

impl<'a, T, D> Drop for Packer<'a, T, D> {
    fn drop(&mut self) {
        todo!();
        //self.flush();
    }
}

//impl TranscriberHandle {
//    pub fn new<D>(model_definition: D) -> Result<Self, D::Error>
//    where
//        D: ModelDefinition<Model = T>,
//    {
//        let (ctrl_tx, ctrl_rx) = channel::<CtrlMsg<T::Data>>();
//
//        let common_model_params = model_definition.common_params();
//        let mut model: T = model_definition.try_into()?;
//
//        let jh = thread::spawn(move || {
//            while let Ok(command) = ctrl_rx.recv() {
//                match command {
//                    CtrlMsg::StartStream { data_rx, string_tx } => {
//                        model.register_sender(0);
//                        while let Ok(msg) = data_rx.recv() {
//                            match msg {
//                                DataMsg::Data(mut data) => {
//                                    let res = model.transcribe(&mut data, 0);
//                                    if string_tx.send(DataMsg::Data(res)).is_err() {
//                                        break;
//                                    };
//                                }
//                                DataMsg::Error(err) => {
//                                    if let Ok(DataMsg::Data(mut data)) = data_rx.recv() {
//                                        let res = model.transcribe(&mut data, 0);
//                                        let _ = string_tx.send(DataMsg::Data(res));
//                                    };
//                                    let _ = string_tx.send(DataMsg::Error(err));
//                                    break;
//                                }
//                            }
//                        }
//                        model.drop_sender(0);
//                    }
//                }
//            }
//        });
//
//        Ok(Self {
//            stream: None,
//            ctrl_tx,
//            common_model_params,
//            model_phantom: PhantomData,
//            jh,
//        })
//    }
//
//    pub fn start_stream(
//        &mut self,
//        mic_settings: MicSettings,
//    ) -> Result<TranscriberStream, StartError> {
//        if self.stream.is_some() {
//            drop(Option::take(&mut self.stream));
//        };
//
//        let (data_tx, data_rx) = channel();
//        let (string_tx, string_rx) = channel();
//
//        self.ctrl_tx
//            .send(CtrlMsg::StartStream { data_rx, string_tx })
//            .map_err(|_| StartError::TranscriberPaniced)?;
//
//        let stream = self.create_stream(mic_settings, data_tx)?;
//        stream.play()?;
//        self.stream = Some(stream);
//
//        Ok(TranscriberStream(string_rx))
//    }
//
//    pub fn stop_stream(&mut self) {
//        drop(Option::take(&mut self.stream));
//    }
//
//    pub fn join(self) -> Result<(), Box<dyn std::any::Any + Send + 'static>> {
//        drop(self.stream);
//        drop(self.ctrl_tx);
//
//        self.jh.join()
//    }
//}

#[derive(Debug, Error)]
pub enum StartError {
    #[error("The transcriber has paniced, call join() to see why")]
    TranscriberPaniced,
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

#[derive(Debug, Error)]
pub enum ReceiveError {
    #[error("The stream is not currently running, call start to start it")]
    TranscriberIdle,
    #[error(transparent)]
    TranscriptionError(#[from] cpal::StreamError),
}

//impl<T> Transcriber<T>
//where
//    T: Model,
//{
//    fn create_stream(
//        &self,
//        mic_settings: MicSettings,
//        data_tx: mpsc::Sender<DataMsg<Box<[T::Data]>>>,
//    ) -> Result<Stream, StartError> {
//        let host = cpal::default_host();
//
//        let device = match mic_settings.selected_device {
//            Some(ref selected_device) => match host.input_devices()?.find(|device| {
//                device
//                    .name()
//                    .map(|device_name| device_name == *selected_device)
//                    .unwrap_or(false)
//            }) {
//                Some(x) => Some(x),
//                None => match mic_settings.on_error {
//                    mic::OnError::Error => return Err(StartError::SelectedDeviceNotFound),
//                    mic::OnError::TryDefault => host.default_input_device(),
//                },
//            },
//            None => host.default_input_device(),
//        }
//        .ok_or(StartError::DeviceError)?;
//
//        let mut input_conf = device
//            .supported_input_configs()?
//            .collect::<Vec<SupportedStreamConfigRange>>();
//        input_conf.sort_by(|lhs, rhs| Self::cmp_config(lhs, rhs));
//
//        loop {
//            let Some(config) = input_conf.pop() else {
//                break Err(StartError::NoConfigFound);
//            };
//            let sample_format = config.sample_format();
//            let config = config
//                .try_with_sample_rate(SampleRate(T::SAMPLE_RATE))
//                .unwrap_or_else(|| config.with_max_sample_rate())
//                .config();
//
//            let msl = self.common_model_params.max_sample_len;
//
//            break Ok(match sample_format {
//                cpal::SampleFormat::I8 => parse_data!(i8, device, config, data_tx, msl),
//                cpal::SampleFormat::I16 => parse_data!(i16, device, config, data_tx, msl),
//                cpal::SampleFormat::I32 => parse_data!(i32, device, config, data_tx, msl),
//                cpal::SampleFormat::I64 => parse_data!(i64, device, config, data_tx, msl),
//                cpal::SampleFormat::U8 => parse_data!(u8, device, config, data_tx, msl),
//                cpal::SampleFormat::U16 => parse_data!(u16, device, config, data_tx, msl),
//                cpal::SampleFormat::U32 => parse_data!(u32, device, config, data_tx, msl),
//                cpal::SampleFormat::U64 => parse_data!(u64, device, config, data_tx, msl),
//                cpal::SampleFormat::F32 => parse_data!(f32, device, config, data_tx, msl),
//                cpal::SampleFormat::F64 => parse_data!(f64, device, config, data_tx, msl),
//                _ => continue,
//            });
//        }
//    }
//}
