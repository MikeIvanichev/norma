use candle_core::Tensor;
use serde::{de::DeserializeOwned, Serialize};
use std::sync::mpsc::{Receiver, Sender};

use crate::{dtype::DType, private::Sealed};

pub mod whisper;

pub trait ModelDefinition: Serialize + DeserializeOwned + PartialEq + Sealed {
    const SAMPLE_RATE: u32;
    type DType: DType;

    /// This function should take the sample and convert it to a Tensor while also preforming any
    /// modifications required by the mode. e.g. changing DType or apllying stride.
    ///
    /// There are however several ganenties applied to the data passed to this fn:
    /// - The data will have the sample rate defined in [`SAMPLE_RATE`]
    /// - The data will have only one channel
    fn pack_data<T>(&self, tx: Sender<Tensor>) -> impl FnMut(T) + Send + 'static
    where
        T: IntoIterator<Item = Self::DType>;

    fn run(&self, rx: Receiver<Tensor>, tx: Sender<String>) -> impl FnOnce() + Send + 'static;
}
