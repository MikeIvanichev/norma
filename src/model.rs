use std::time::Duration;

use serde::{de::DeserializeOwned, Serialize};

use crate::dtype::DType;

pub mod dummy;
pub mod whisper;

pub enum ModelInput<T> {
    Data(T),
    ClearContext,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CommonModelParams {
    pub max_sample_len: usize,
    pub data_buffer: usize,
    pub string_buffer: usize,
}

pub trait ModelDefinition:
    TryInto<Self::Model> + Serialize + DeserializeOwned + PartialEq + Sync + Send
{
    type Model;

    fn common_params(&self) -> CommonModelParams;
}

pub trait Model: Send + 'static {
    type Data: DType;
    const SAMPLE_RATE: u32;

    fn clear_context(&mut self);
    fn transcribe(&mut self, data: &mut Vec<Self::Data>) -> String;
}
