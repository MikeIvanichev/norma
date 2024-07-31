//! [Whisper](https://github.com/openai/whisper) is a Speech-to-Text (STT) model developed by [OpenAI](https://openai.com/).
//!
//! This implementation of Whisper is powered by [candle](https://github.com/huggingface/candle).
//! Long-form transcription is fully implemented, as described in [the whisper papper](https://arxiv.org/abs/2212.04356)
//!
//! Whisper comes with multiple wheights/checkpoints
//! that differ by inference speed, accuracy, voabulary and supported languages.
//! Refer to the below table for specifics.
//!
//! | Model           | Params (M) | Relative speed | Short-Form WER | Long-Form WER | Vocab | Languages |
//! |-----------------|------------|----------------|----------------|---------------|-------|-----------|
//! | QuantizedTiny   |            |                |                |               | V1    | All       |
//! | Tiny            | 39         | 5.75           |                |               | V1    | All       |
//! | Base            | 74         | 4.6            |                |               | V1    | All       |
//! | Small           | 244        | 2.4            |                |               | V1    | All       |
//! | Medium          | 769        | 1.35           |                |               | V1    | All       |
//! | Large           | 1550       | 1.0            |                |               | V1    | All       |
//! | QuantizedTinyEn |            |                |                |               | EnV1  | English   |
//! | TinyEn          | 39         | 5.75           | 18.9           | 18.9          | EnV1  | English   |
//! | BaseEn          | 74         | 4.6            | 14.3           | 15.7          | EnV1  | English   |
//! | SmallEn         | 244        | 2.4            | 10.8           | 14.7          | EnV1  | English   |
//! | MediumEn        | 769        | 1.35           | 9.5            | 12.3          | EnV1  | English   |
//! | LargeV2         | 1550       | 1.0            | 9.1            | 11.7          | V1    | All       |
//! | LargeV3         | 1550       | 1.0            | 8.4            | 11.0          | V2    | All       |
//! | DistilLargeEnV3 | 756        | 6.3            | 9.7            | 10.8          | V2    | English   |
//! | DistilLargeEnV2 | 756        | 5.8            | 10.1           | 11.6          | V1    | English   |
//! | DistilMediumEn  | 394        | 6.8            | 11.1           | 12.4          | V1    | English   |
//!
//! # Models
//!
//! ## Model Types
//!
//! ### [Distil-Whisper](https://github.com/huggingface/distil-whisper)
//!
//! ### Quantized

mod languages;
pub use languages::Language;

mod model;
use model::LanguageState;
pub use model::Model;
pub use model::TranscriberError;
use model::Type;

pub mod monolingual;

pub mod multilingual;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokenizers::Tokenizer;

use super::CMPError;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum VocabVersion {
    V1,
    V2,
    EnV1,
    EnV2,
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("Failed to get token ID for: {0}")]
    TokenId(String),
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error(transparent)]
    HubAPI(#[from] hf_hub::api::sync::ApiError),
    #[error(transparent)]
    AsyncHubAPI(#[from] hf_hub::api::tokio::ApiError),
    #[error("Failed to laod the tokenizer becouse: {0}")]
    LoadTokenizer(Box<dyn std::error::Error>),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    ParseJson(#[from] serde_json::Error),
    #[error("Unexpected number of mel bins (num_mel_bins), got: {0}")]
    MelBins(usize),
    #[error("The respnsivness must be over 1 second and under 30")]
    Respnsivness,
    #[error(transparent)]
    CMPError(#[from] CMPError),
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32, Error> {
    tokenizer
        .token_to_id(token)
        .ok_or(Error::TokenId(token.to_owned()))
}
