mod languages;
pub use languages::Language;

mod model;
use model::LanguageState;
pub use model::Model;
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
