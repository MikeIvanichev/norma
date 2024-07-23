use std::{fmt::Debug, future::Future};

use serde::{Deserialize, Deserializer, Serialize};
use thiserror::Error;

use crate::dtype::DType;

pub mod whisper;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum SelectedDevice {
    #[default]
    Cpu,
    Cuda(usize),
    Metal,
}

impl SelectedDevice {
    pub(crate) fn into_cpal_device(self) -> Result<candle_core::Device, candle_core::Error> {
        match self {
            SelectedDevice::Cpu => Ok(candle_core::Device::Cpu),
            SelectedDevice::Cuda(n) => candle_core::Device::new_cuda(n),
            SelectedDevice::Metal => candle_core::Device::new_metal(0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CommonModelParams {
    /// The max number of samples to buffer before sending to the Transcriber
    max_chunk_len: usize,
    /// The buffer size of the channel between the Mic and the Transcriber, in chunks
    data_buffer_size: usize,
    /// The buffer size of the channel between the Transcriber and the String Receiver
    string_buffer_size: usize,
}

fn de_common_model_params<'de, D>(deserializer: D) -> Result<CommonModelParams, D::Error>
where
    D: Deserializer<'de>,
{
    let params = CommonModelParams::deserialize(deserializer)?;

    if params.data_buffer_size < 3 {
        return Err(serde::de::Error::invalid_value(
            serde::de::Unexpected::StructVariant,
            &"a data buffer size greater then 2",
        ));
    }

    if params.string_buffer_size < 3 {
        return Err(serde::de::Error::invalid_value(
            serde::de::Unexpected::StructVariant,
            &"a string buffer size greater then 2",
        ));
    }

    Ok(params)
}

#[derive(Debug, Error)]
pub enum CMPError {
    #[error("The Data buffer size must be at least 2")]
    DataBufSize,
    #[error("The String buffer size must be at least 2")]
    StringBufSize,
}

impl CommonModelParams {
    pub fn new(
        max_chunk_len: usize,
        data_buffer_size: usize,
        string_buffer_size: usize,
    ) -> Result<Self, CMPError> {
        Ok(Self {
            max_chunk_len,
            data_buffer_size: data_buffer_size + 2,
            string_buffer_size: string_buffer_size + 2,
        })
    }

    pub fn max_chunk_len(&self) -> usize {
        self.max_chunk_len
    }

    pub fn data_buffer_size(&self) -> usize {
        self.data_buffer_size
    }

    pub fn string_buffer_size(&self) -> usize {
        self.string_buffer_size
    }

    pub fn set_max_chunk_len(&mut self, max_chunk_len: usize) {
        self.max_chunk_len = max_chunk_len;
    }

    pub fn set_data_buffer_size(&mut self, data_buffer_size: usize) -> Result<(), CMPError> {
        self.data_buffer_size = data_buffer_size + 2;
        Ok(())
    }

    pub fn set_string_buffer_size(&mut self, string_buffer_size: usize) -> Result<(), CMPError> {
        self.string_buffer_size = string_buffer_size + 2;
        Ok(())
    }
}
