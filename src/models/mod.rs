use std::{fmt::Debug, future::Future};

use tracing::warn;

use crate::dtype::DType;

#[cfg(feature = "whisper")]
pub mod whisper;

#[cfg(feature = "_mock")]
pub mod mock;

pub trait ModelDefinition {
    type Model;
    type Error: std::error::Error;

    fn common_params(&self) -> &CommonModelParams;

    fn try_to_model(self) -> impl Future<Output = Result<Self::Model, Self::Error>> + Send;

    fn blocking_try_to_model(self) -> Result<Self::Model, Self::Error>;
}

pub trait Model: Send + 'static {
    type Data: DType;
    type Error: std::error::Error + Send + 'static;
    const SAMPLE_RATE: u32;

    fn transcribe(
        &mut self,
        data: &mut Vec<Self::Data>,
        final_chunk: bool,
    ) -> Result<String, Self::Error>;
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SelectedDevice {
    #[default]
    Cpu,
    Cuda(usize),
    Metal,
}

impl SelectedDevice {
    #[cfg(feature = "whisper")]
    pub(crate) fn into_cpal_device(self) -> Result<candle_core::Device, candle_core::Error> {
        match self {
            SelectedDevice::Cpu => Ok(candle_core::Device::Cpu),
            SelectedDevice::Cuda(n) => candle_core::Device::new_cuda(n),
            // For now i dont see a point in letting the user set ordinal, since its
            // rather unlikely for apple to release a > 1 gpu device.
            SelectedDevice::Metal => candle_core::Device::new_metal(0),
        }
    }
}

/// It would be *insanely* wastefull to have a size below this
const MIN_CHUNK_LEN: usize = 100;
/// As we are using ThingBuff for our data channel we need to have a size of >= 2
const MIN_DATA_BUF_SIZE: usize = 2;
/// Tokio mpsc channels will panic if size is < 1, so make sure its above one
const MIN_STRING_BUF_SIZE: usize = 1;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CommonModelParams {
    /// The max number of samples to buffer before sending to the Transcr ber
    max_chunk_len: usize,
    /// The buffer size of the channel between the Input and the Transcriber, in chunks
    data_buffer_size: usize,
    /// The buffer size of the channel between the Transcriber and the String Receiver
    string_buffer_size: usize,
}

impl CommonModelParams {
    pub fn new(max_chunk_len: usize, data_buffer_size: usize, string_buffer_size: usize) -> Self {
        Self {
            max_chunk_len: max_chunk_len.max(MIN_CHUNK_LEN),
            // since we are using Thingbuff the actual buff size would be n - 2
            data_buffer_size: data_buffer_size + 2,
            string_buffer_size: string_buffer_size.max(MIN_STRING_BUF_SIZE),
        }
    }

    pub fn max_chunk_len(&self) -> usize {
        if self.max_chunk_len < MIN_CHUNK_LEN {
            warn!(max_chunk_len = self.max_chunk_len,
                    MIN_CHUNK_LEN,
                    "The chunk length is too small, it should not be possible to set it to this value. Returning MIN_CHUBK_LEN instead.");
            return MIN_CHUNK_LEN;
        }

        self.max_chunk_len
    }

    pub fn data_buffer_size(&self) -> usize {
        self.data_buffer_size
    }

    pub fn string_buffer_size(&self) -> usize {
        self.string_buffer_size
    }

    pub fn set_max_chunk_len(&mut self, max_chunk_len: usize) {
        self.max_chunk_len = max_chunk_len.max(MIN_CHUNK_LEN);
    }

    pub fn set_data_buffer_size(&mut self, data_buffer_size: usize) {
        // since we are using Thingbuff the actual buff size would be n - 2
        self.data_buffer_size = data_buffer_size + 2;
    }

    pub fn set_string_buffer_size(&mut self, string_buffer_size: usize) {
        self.string_buffer_size = string_buffer_size.max(MIN_STRING_BUF_SIZE);
    }
}
