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
    sync::{
        mpsc::{channel, SyncSender},
        Arc,
    },
    thread::{self, JoinHandle},
};

use crate::dtype::DType;

use candle_core::WithDType;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SampleRate, SupportedStreamConfigRange,
};
use mic::MicSettings;
use model::{dummy::DummyDef, CommonModelParams, Model, ModelDefinition, ModelInput};
use thingbuf::{mpsc::blocking::SendRef, recycling::WithCapacity};
use thiserror::Error;

macro_rules! parse_data {
    ($t:ty, $device:ident, $config:ident, $tx: ident, $error_tx: ident, $msl: ident) => {{
        use dasp_frame::Frame;
        use dasp_signal::Signal;

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
                let _ = $error_tx.send(err);
            },
            None,
        )?;
        stream.play()?;
        stream
    }};
}

pub(crate) use parse_data;
use tracing::warn;

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
                mem::swap(send_ref.deref_mut(), &mut self.buff);
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
pub enum TranscriberError {
    #[error(transparent)]
    StreamError(#[from] cpal::StreamError),
}
