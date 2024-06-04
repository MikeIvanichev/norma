use candle_core::Tensor;
use serde::{Deserialize, Serialize};

use crate::{private, ModelDefinition};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct WhisperSettings {
    model: Model,
    quantized: bool,
    task: Task,
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
