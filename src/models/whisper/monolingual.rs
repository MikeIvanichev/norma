use std::time::Duration;

use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, Config};
use hf_hub::{api::sync, Repo, RepoType};
use rand::SeedableRng;
use tokenizers::Tokenizer;
use tracing::{instrument, Level};

use crate::models::{CommonModelParams, ModelDefinition, SelectedDevice};

use super::{languages::Language, token_id, LanguageState, VocabVersion};

/// Monolingual Whisper checkpoints of different sizes.
///
/// | ModelType       | Language | Vocab |
/// |-----------------|----------|-------|
/// | QuantizedTinyEn | English  | EnV1  |
/// | TinyEn          | English  | EnV1  |
/// | BaseEn          | English  | EnV1  |
/// | SmallEn         | English  | EnV1  |
/// | MediumEn        | English  | EnV1  |
/// | DistilMediumEn  | English  | V1    |
/// | DistilLargeEnV2 | English  | V1    |
/// | DistilLargeEnV3 | English  | V2    |
///
/// You can use [MultiAsMono][ModelType::MultiAsMono] to treat a [Multilingual][super::multilingual::ModelType] model as monolingual.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelType {
    QuantizedTinyEn,
    TinyEn,
    BaseEn,
    SmallEn,
    MediumEn,
    DistilMediumEn,
    DistilLargeEnV2,
    #[default]
    DistilLargeEnV3,
    MultiAsMono {
        model: super::multilingual::ModelType,
        lang: Language,
    },
}

impl ModelType {
    pub fn id(&self) -> &str {
        match self {
            ModelType::QuantizedTinyEn => "lmz/candle-whisper",
            ModelType::TinyEn => "openai/whisper-tiny.en",
            ModelType::BaseEn => "openai/whisper-base.en",
            ModelType::SmallEn => "openai/whisper-small.en",
            ModelType::MediumEn => "openai/whisper-medium.en",
            ModelType::DistilMediumEn => "distil-whisper/distil-medium.en",
            ModelType::DistilLargeEnV2 => "distil-whisper/distil-large-v2",
            ModelType::DistilLargeEnV3 => "distil-whisper/distil-large-v3",
            ModelType::MultiAsMono { model, .. } => model.id(),
        }
    }

    pub fn rev(&self) -> &str {
        match self {
            ModelType::QuantizedTinyEn
            | ModelType::MediumEn
            | ModelType::DistilMediumEn
            | ModelType::DistilLargeEnV2
            | ModelType::DistilLargeEnV3 => "main",
            ModelType::TinyEn => "refs/pr/15",
            ModelType::BaseEn => "refs/pr/13",
            ModelType::SmallEn => "refs/pr/10",
            ModelType::MultiAsMono { model, .. } => model.rev(),
        }
    }

    pub fn quantized_ext(&self) -> Option<&str> {
        match self {
            ModelType::QuantizedTinyEn => Some("tiny-en"),
            ModelType::MultiAsMono { model, .. } => model.quantized_ext(),
            _ => None,
        }
    }

    pub fn language(&self) -> Language {
        match self {
            ModelType::QuantizedTinyEn
            | ModelType::TinyEn
            | ModelType::BaseEn
            | ModelType::SmallEn
            | ModelType::MediumEn
            | ModelType::DistilMediumEn
            | ModelType::DistilLargeEnV2
            | ModelType::DistilLargeEnV3 => Language::English,
            ModelType::MultiAsMono { lang, .. } => lang.clone(),
        }
    }

    pub fn vocab_version(&self) -> super::VocabVersion {
        match self {
            ModelType::QuantizedTinyEn
            | ModelType::TinyEn
            | ModelType::BaseEn
            | ModelType::SmallEn
            | ModelType::MediumEn => VocabVersion::EnV1,
            ModelType::DistilMediumEn | ModelType::DistilLargeEnV2 => VocabVersion::V1,
            ModelType::DistilLargeEnV3 => VocabVersion::V2,
            ModelType::MultiAsMono { model, .. } => model.vocab_version(),
        }
    }
}

/// The definition (config) of a monolingual whisper model.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Definition {
    model: ModelType,
    device: SelectedDevice,
    common_params: CommonModelParams,
}

impl Definition {
    #[instrument(ret(level = Level::DEBUG))]
    pub fn new(model: ModelType, device: SelectedDevice) -> Self {
        Self {
            model,
            device,
            common_params: CommonModelParams::new(m::SAMPLE_RATE * 25, 3, 3),
        }
    }

    /// With a valid range of [1..=30] seconds,
    /// this sets how often the model should attempt to decode the recorded sample.
    /// Smaller numbers will result in closer to real-time transcription,
    /// while requiring more compute.
    ///
    /// For example, if set to 10 seconds:
    ///
    /// Every 10 seconds the model will attempt to transcribe the recorded segment.
    /// If nothing is successfully transcribed, the model retains the recorded segment and
    /// combines it with the next segment.
    /// In this case creating a 20-second segment, which it will then attempt to transcribe.
    /// This continues until something is successfully transcribed,
    /// or a full segment of >= 30 seconds is accumilated.
    /// In which case regardless of the outcome, the result will be returned.
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

    /// Sets the size of the buffer that stores recorded segments before they are transcribed.
    /// Mesured in number of segments.
    ///
    /// When the buffer is full, new segments will be droped.
    #[instrument(skip(self))]
    pub fn set_data_buffer_size(&mut self, size: usize) {
        self.common_params.set_data_buffer_size(size);
    }

    /// Sets the size of the buffer that stores transcribed segments (Strings).
    ///
    /// When the buffer is full, new segments will be droped.
    #[instrument(skip(self))]
    pub fn set_string_buffer_size(&mut self, size: usize) {
        self.common_params.set_string_buffer_size(size);
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
        let device = self.device.into_cpal_device()?;

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

        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok())
            .ok_or(super::Error::TokenId(m::NO_SPEECH_TOKENS.join(" nor ")))?;
        let task_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let lang_token = token_id(&tokenizer, self.model.language().token())?;
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;

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
            lang: LanguageState::ConstLang(lang_token),
        })
    }

    #[instrument(level = Level::DEBUG, err(Display))]
    fn blocking_try_to_model(self) -> Result<Self::Model, Self::Error> {
        let device = self.device.into_cpal_device()?;

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

        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok())
            .ok_or(super::Error::TokenId(m::NO_SPEECH_TOKENS.join(" nor ")))?;
        let task_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let lang_token = token_id(&tokenizer, self.model.language().token())?;

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
            lang: LanguageState::ConstLang(lang_token),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn download_and_validate_model(model: ModelType) {
        let (config_file, tokenizer_file, _weights_file) = {
            let api = sync::Api::new().unwrap();

            let repo = api.repo(Repo::with_revision(
                model.id().to_string(),
                RepoType::Model,
                model.rev().to_string(),
            ));

            if let Some(ext) = model.quantized_ext() {
                (
                    repo.get(&format!("config-{ext}.json")).unwrap(),
                    repo.get(&format!("tokenizer-{ext}.json")).unwrap(),
                    repo.get(&format!("model-{ext}-q80.gguf")).unwrap(),
                )
            } else {
                (
                    repo.get("config.json").unwrap(),
                    repo.get("tokenizer.json").unwrap(),
                    repo.get("model.safetensors").unwrap(),
                )
            }
        };

        let _config: Config =
            serde_json::from_str(&std::fs::read_to_string(config_file).unwrap()).unwrap();
        let _tokenizer = Tokenizer::from_file(tokenizer_file).unwrap();
    }

    #[test]
    #[ignore = "Not enough storage on github action host"]
    fn download_quantized_tiny_en() {
        download_and_validate_model(ModelType::QuantizedTinyEn);
    }

    #[test]
    #[ignore = "Not enough storage on github action host"]
    fn download_tiny_en() {
        download_and_validate_model(ModelType::TinyEn);
    }

    #[test]
    #[ignore = "Not enough storage on github action host"]
    fn download_base_en() {
        download_and_validate_model(ModelType::BaseEn);
    }

    #[test]
    #[ignore = "Not enough storage on github action host"]
    fn download_small_en() {
        download_and_validate_model(ModelType::SmallEn);
    }

    #[test]
    #[ignore = "Not enough storage on github action host"]
    fn download_medium_en() {
        download_and_validate_model(ModelType::MediumEn);
    }

    #[test]
    #[ignore = "Not enough storage on github action host"]
    fn download_distil_medium_en() {
        download_and_validate_model(ModelType::DistilMediumEn);
    }

    #[test]
    #[ignore = "Not enough storage on github action host"]
    fn download_distil_large_en_v2() {
        download_and_validate_model(ModelType::DistilLargeEnV2);
    }

    #[test]
    #[ignore = "Not enough storage on github action host"]
    fn download_distil_large_en_v3() {
        download_and_validate_model(ModelType::DistilLargeEnV3);
    }
}
