use candle_core::Tensor;
use cpal::SizedSample;
use serde::{Deserialize, Serialize};

use crate::{private, ModelDefinition};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct WhisperSettings {
    model: Model,
    quantized: bool,
    task: Task,
}

impl Into<Whisper> for WhisperSettings {
    fn into(self) -> Whisper {
        todo!()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Device {
    #[default]
    Cpu,
    Cuda(usize),
    Metal(usize),
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub enum Model {
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub enum Task {
    #[default]
    Transcribe,
    Translate,
}

// -------------------------------

struct Whisper {}
