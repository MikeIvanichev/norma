use std::{fmt::Debug, future::Future};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::dtype::DType;

pub mod whisper;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum SelectedDevice {
    #[default]
    Cpu,
    Cuda(usize),
    Metal(usize),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CommonModelParams {
    /// The max number of samples to buffer before sending to the Transcriber
    pub(crate) max_chunk_len: usize,
    /// The buffer size of the channel between the Mic and the Transcriber, in chunks
    pub(crate) data_buffer_size: usize,
    /// The buffer size of the channel between the Transcriber and the String Receiver
    pub(crate) string_buffer_size: usize,
}

pub trait ModelDefinition: Debug + Serialize + DeserializeOwned + PartialEq + Sync + Send {
    type Model;
    type Error: std::error::Error;

    fn common_params(&self) -> CommonModelParams;

    fn try_to_model(self) -> impl Future<Output = Result<Self::Model, Self::Error>> + Send;

    fn blocking_try_to_model(self) -> Result<Self::Model, Self::Error>;
}

pub trait Model: Send + 'static {
    type Data: DType;
    const SAMPLE_RATE: u32;

    fn transcribe(&mut self, data: &mut Vec<Self::Data>, final_chunk: bool) -> Option<String>;
}
