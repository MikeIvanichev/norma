use rand::SeedableRng;
use std::time::Duration;
use strum::IntoEnumIterator;
use tracing::{instrument, Level};

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, Config};
use hf_hub::{api::sync, Repo, RepoType};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::models::{CommonModelParams, ModelDefinition, SelectedDevice};

use super::{model::LanguageState, token_id, Language, VocabVersion};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Task {
    #[default]
    // Transcribes the test in the detected language.
    Transcribe,
    /// Translates from the detected language into English.
    Translate,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ModelType {
    QuantizedTiny,
    Tiny,
    Base,
    Small,
    #[default]
    Medium,
    Large,
    LargeV2,
    LargeV3,
}

impl ModelType {
    pub fn id(&self) -> &str {
        match self {
            ModelType::QuantizedTiny => "openai/whisper-tiny",
            ModelType::Tiny => "openai/whisper-tiny",
            ModelType::Base => "openai/whisper-base",
            ModelType::Small => "openai/whisper-small",
            ModelType::Medium => "openai/whisper-medium",
            ModelType::Large => "openai/whisper-large",
            ModelType::LargeV2 => "openai/whisper-large-v2",
            ModelType::LargeV3 => "openai/whisper-large-v3",
        }
    }

    pub fn rev(&self) -> &str {
        match self {
            ModelType::QuantizedTiny
            | ModelType::Tiny
            | ModelType::Small
            | ModelType::Medium
            | ModelType::LargeV3 => "main",
            ModelType::Base => "refs/pr/22",
            ModelType::Large => "refs/pr/36",
            ModelType::LargeV2 => "refs/pr/57",
        }
    }

    pub fn quantized_ext(&self) -> Option<&str> {
        match self {
            ModelType::QuantizedTiny => Some("tiny"),
            _ => None,
        }
    }

    pub fn vocab_version(&self) -> VocabVersion {
        match self {
            ModelType::QuantizedTiny
            | ModelType::Tiny
            | ModelType::Base
            | ModelType::Small
            | ModelType::Medium
            | ModelType::Large
            | ModelType::LargeV2 => VocabVersion::V1,
            ModelType::LargeV3 => VocabVersion::V2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Definition {
    model: ModelType,
    device: SelectedDevice,
    task: Task,
    common_params: CommonModelParams,
}

impl Definition {
    #[instrument(ret(level = Level::DEBUG))]
    pub fn new(model: ModelType, device: SelectedDevice, task: Task) -> Self {
        Self {
            model,
            device,
            task,
            common_params: CommonModelParams::new(m::SAMPLE_RATE * 25, 3, 3).unwrap(),
        }
    }

    #[instrument(skip(self), err(Display, level = Level::DEBUG))]
    pub fn set_responsiveness(&mut self, period: Duration) -> Result<(), super::Error> {
        let period = period.as_millis();
        if (1_000..=30_000).contains(&period) {
            self.common_params
                .set_max_chunk_len((m::SAMPLE_RATE * period as usize) / 1000);
            Ok(())
        } else {
            Err(super::Error::Respnsivness)
        }
    }

    #[instrument(skip(self), err(Display, level = Level::DEBUG))]
    pub fn set_data_buffer_size(&mut self, size: usize) -> Result<(), super::Error> {
        self.common_params.set_data_buffer_size(size)?;
        Ok(())
    }

    #[instrument(skip(self), err(Display, level = Level::DEBUG))]
    pub fn set_string_buffer_size(&mut self, size: usize) -> Result<(), super::Error> {
        self.common_params.set_string_buffer_size(size)?;
        Ok(())
    }
}

impl ModelDefinition for Definition {
    type Model = super::Model;

    type Error = super::Error;

    fn common_params(&self) -> &CommonModelParams {
        &self.common_params
    }

    #[instrument(level = Level::DEBUG, err(Display))]
    async fn try_to_model(self) -> Result<Self::Model, Self::Error> {
        let device = match self.device {
            SelectedDevice::Cpu => Device::Cpu,
            SelectedDevice::Cuda(n) => Device::new_cuda(n)?,
            SelectedDevice::Metal(n) => Device::new_metal(n)?,
        };

        let (config_file, tokenizer_file, weights_file) = {
            let api = hf_hub::api::tokio::Api::new()?;

            let repo = api.repo(Repo::with_revision(
                self.model.id().to_string(),
                RepoType::Model,
                self.model.rev().to_string(),
            ));

            if let Some(ext) = self.model.quantized_ext() {
                (
                    repo.get(&format!("config-{ext}.json")).await?,
                    repo.get(&format!("tokenizer-{ext}.json")).await?,
                    repo.get(&format!("model-{ext}-q80.gguf")).await?,
                )
            } else {
                (
                    repo.get("config.json").await?,
                    repo.get("tokenizer.json").await?,
                    repo.get("model.safetensors").await?,
                )
            }
        };

        let config: Config = serde_json::from_str(&tokio::fs::read_to_string(config_file).await?)?;
        let tokenizer =
            Tokenizer::from_file(tokenizer_file).map_err(|err| super::Error::LoadTokenizer(err))?;

        let mel_bytes = match config.num_mel_bins {
            80 => Ok(include_bytes!("./whisper_mel_bytes/80.bytes").as_slice()),
            128 => Ok(include_bytes!("./whisper_mel_bytes/128.bytes").as_slice()),
            nmel => Err(super::Error::MelBins(nmel)),
        }?;

        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];

        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            mel_bytes,
            &mut mel_filters,
        );

        let model = if self.model.quantized_ext().is_some() {
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &weights_file,
                &device,
            )?;
            super::Type::Quantized(m::quantized_model::Whisper::load(&vb, config.clone())?)
        } else {
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[weights_file], m::DTYPE, &device)? };
            super::Type::Normal(m::model::Whisper::load(&vb, config.clone())?)
        };

        let task_token = match self.task {
            Task::Transcribe => token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?,
            Task::Translate => token_id(&tokenizer, m::TRANSLATE_TOKEN)?,
        };
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok())
            .ok_or(super::Error::TokenId(m::NO_SPEECH_TOKENS.join(" nor ")))?;
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;

        let language_tokens_tensor = Language::iter()
            .map(|lang| token_id(&tokenizer, lang.token()))
            .collect::<Result<Vec<_>, _>>()?;
        let language_tokens_tensor = Tensor::new(language_tokens_tensor.as_slice(), &device)?;

        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i) || i == no_timestamps_token {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), &device)?;

        let supress_non_timestamps: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if i > no_timestamps_token {
                    0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        let supress_non_timestamps = Tensor::new(supress_non_timestamps.as_slice(), &device)?;

        let supress_timestamps: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if i > no_timestamps_token {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let supress_timestamps = Tensor::new(supress_timestamps.as_slice(), &device)?;

        let zero_sec = token_id(&tokenizer, "<|0.00|>")?;
        let one_sec = token_id(&tokenizer, "<|1.00|>")?;
        let first_token_supress: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if i < zero_sec || i > one_sec {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let first_token_supress = Tensor::new(first_token_supress.as_slice(), &device)?;

        Ok(Self::Model {
            model,
            buf: Vec::with_capacity(self.common_params.max_chunk_len()),
            config,
            device,
            mel_filters,
            tokenizer,
            rng: rand::rngs::StdRng::from_entropy(),
            task_token,
            suppress_tokens,
            supress_non_timestamps,
            supress_timestamps,
            first_token_supress,
            sot_token,
            eot_token,
            no_speech_token,
            no_timestamps_token,
            lang: LanguageState::Detect {
                language_token: None,
                language_tokens_tensor,
            },
        })
    }

    #[instrument(level = Level::DEBUG, err(Display))]
    fn blocking_try_to_model(self) -> Result<Self::Model, Self::Error> {
        let device = match self.device {
            SelectedDevice::Cpu => Device::Cpu,
            SelectedDevice::Cuda(n) => Device::new_cuda(n)?,
            SelectedDevice::Metal(n) => Device::new_metal(n)?,
        };

        let (config_file, tokenizer_file, weights_file) = {
            let api = sync::Api::new()?;

            let repo = api.repo(Repo::with_revision(
                self.model.id().to_string(),
                RepoType::Model,
                self.model.rev().to_string(),
            ));

            if let Some(ext) = self.model.quantized_ext() {
                (
                    repo.get(&format!("config-{ext}.json"))?,
                    repo.get(&format!("tokenizer-{ext}.json"))?,
                    repo.get(&format!("model-{ext}-q80.gguf"))?,
                )
            } else {
                (
                    repo.get("config.json")?,
                    repo.get("tokenizer.json")?,
                    repo.get("model.safetensors")?,
                )
            }
        };

        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_file)?)?;
        let tokenizer =
            Tokenizer::from_file(tokenizer_file).map_err(|err| super::Error::LoadTokenizer(err))?;

        let mel_bytes = match config.num_mel_bins {
            80 => Ok(include_bytes!("./whisper_mel_bytes/80.bytes").as_slice()),
            128 => Ok(include_bytes!("./whisper_mel_bytes/128.bytes").as_slice()),
            nmel => Err(super::Error::MelBins(nmel)),
        }?;

        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];

        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            mel_bytes,
            &mut mel_filters,
        );

        let model = if self.model.quantized_ext().is_some() {
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &weights_file,
                &device,
            )?;
            super::Type::Quantized(m::quantized_model::Whisper::load(&vb, config.clone())?)
        } else {
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[weights_file], m::DTYPE, &device)? };
            super::Type::Normal(m::model::Whisper::load(&vb, config.clone())?)
        };

        let task_token = match self.task {
            Task::Transcribe => token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?,
            Task::Translate => token_id(&tokenizer, m::TRANSLATE_TOKEN)?,
        };
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok())
            .ok_or(super::Error::TokenId(m::NO_SPEECH_TOKENS.join(" nor ")))?;
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;

        let language_tokens_tensor = Language::iter()
            .map(|lang| token_id(&tokenizer, lang.token()))
            .collect::<Result<Vec<_>, _>>()?;
        let language_tokens_tensor = Tensor::new(language_tokens_tensor.as_slice(), &device)?;

        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i) || i == no_timestamps_token {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), &device)?;

        let supress_non_timestamps: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if i > no_timestamps_token {
                    0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        let supress_non_timestamps = Tensor::new(supress_non_timestamps.as_slice(), &device)?;

        let supress_timestamps: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if i > no_timestamps_token {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let supress_timestamps = Tensor::new(supress_timestamps.as_slice(), &device)?;

        let zero_sec = token_id(&tokenizer, "<|0.00|>")?;
        let one_sec = token_id(&tokenizer, "<|1.00|>")?;
        let first_token_supress: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if i < zero_sec || i > one_sec {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let first_token_supress = Tensor::new(first_token_supress.as_slice(), &device)?;

        Ok(Self::Model {
            model,
            buf: Vec::with_capacity(self.common_params.max_chunk_len()),
            config,
            device,
            mel_filters,
            tokenizer,
            rng: rand::rngs::StdRng::from_entropy(),
            task_token,
            suppress_tokens,
            supress_non_timestamps,
            supress_timestamps,
            first_token_supress,
            sot_token,
            eot_token,
            no_speech_token,
            no_timestamps_token,
            lang: LanguageState::Detect {
                language_token: None,
                language_tokens_tensor,
            },
        })
    }
}
