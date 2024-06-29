use serde::{Deserialize, Serialize};

use super::{CommonModelParams, ModelDefinition};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Device {
    #[default]
    Cpu,
    Cuda(usize),
    Metal(usize),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ModelType {
    Tiny,
    TinyEn,
    Base,
    BaseEn,
    Small,
    SmallEn,
    Medium,
    MediumEn,
    Large,
    LargeV2,
    LargeV3,
    DistilMediumEn,
    DistilLargeV2,
    #[default]
    DistilLargeV3,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Task {
    #[default]
    Transcribe,
    Translate,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WhisperSettings {
    model: ModelType,
    quantized: bool,
    task: Task,
    device: Device,
    max_sample_len: usize,
    data_buffer: usize,
    string_buffer: usize,
    stride: usize,
}

impl ModelDefinition for WhisperSettings {
    type Model = Whisper;

    fn common_params(&self) -> CommonModelParams {
        CommonModelParams {
            max_sample_len: self.max_sample_len,
            data_buffer: self.data_buffer,
            string_buffer: self.string_buffer,
        }
    }
}

pub enum Error {}

impl TryInto<Whisper> for WhisperSettings {
    type Error = Error;

    fn try_into(self) -> Result<Whisper, Self::Error> {
        todo!()
    }
}

// -------------------------------

pub struct Whisper {}

impl crate::model::Model for Whisper {
    type Data = f32;

    const SAMPLE_RATE: u32 = 16_000;

    fn clear_context(&mut self) {
        todo!()
    }

    fn transcribe(&mut self, data: &mut Vec<Self::Data>) -> String {
        todo!()
    }
}
