use std::mem;

use candle_core::{Device, IndexOp, Tensor, D};
use candle_nn::ops::softmax;
use rand::distributions::Distribution;
use strum::IntoEnumIterator;
use thiserror::Error;
use tokenizers::Tokenizer;
use tracing::{debug, debug_span, instrument, trace, warn, Level};

use crate::utils::SliceExt;
use candle_transformers::models::whisper::{self as m, audio, Config};

use super::languages::Language;

pub struct Model {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) device: Device,
    // ---
    pub(crate) model: Type,
    pub(crate) config: Config,
    pub(crate) mel_filters: Vec<f32>,
    // ---
    pub(crate) buf: Vec<f32>,
    // ---
    pub(crate) rng: rand::rngs::StdRng,
    // ---
    pub(crate) lang: LanguageState,
    // ---
    pub(crate) suppress_tokens: Tensor,
    pub(crate) supress_non_timestamps: Tensor,
    /// Must supress only timestamps
    pub(crate) supress_timestamps: Tensor,
    /// Must supress all tokens except timestamps [0..1]
    pub(crate) first_token_supress: Tensor,
    // ---
    pub(crate) sot_token: u32,
    pub(crate) eot_token: u32,
    pub(crate) task_token: u32,
    pub(crate) no_speech_token: u32,
    pub(crate) no_timestamps_token: u32,
}

#[derive(Debug, Error)]
#[error(transparent)]
pub struct TranscriberError(#[from] candle_core::Error);

impl crate::models::Model for Model {
    type Data = f32;
    type Error = TranscriberError;

    const SAMPLE_RATE: u32 = 16_000;

    #[instrument(skip(self, data), fields(input_data_len = data.len(), buf_len = self.buf.len()))]
    fn transcribe(
        &mut self,
        data: &mut Vec<Self::Data>,
        final_chunk: bool,
    ) -> Result<String, Self::Error> {
        if self.buf.is_empty() {
            mem::swap(&mut self.buf, data);
        } else {
            self.buf.append(data);
        }

        let mut res = String::new();

        'new_chunk: while !self.buf.is_empty() {
            let slice_len = self.buf.len().min(m::N_SAMPLES);
            let data_slice = &self.buf[..slice_len];

            let _span = debug_span!("Transcribe slice", slice_len).entered();

            let mel = audio::pcm_to_mel(&self.config, data_slice, &self.mel_filters);
            let mel_len = mel.len();
            let mel = Tensor::from_vec(
                mel,
                (
                    1,
                    self.config.num_mel_bins,
                    mel_len / self.config.num_mel_bins,
                ),
                &self.device,
            )
            .unwrap();

            let (_, _, mel_len) = mel.dims3().unwrap();
            let mel = mel.narrow(2, 0, m::N_FRAMES.min(mel_len)).unwrap();

            let Some(dr) = self.decode_with_fallback(&mel)? else {
                self.buf.drain(..slice_len);
                continue;
            };

            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                self.buf.drain(..slice_len);
                continue;
            };

            for tokens in dr.tokens.inclusive_boxed_by(|&token| {
                token > self.no_timestamps_token || token == self.eot_token
            }) {
                let s_timestamp = tokens[0] - self.no_timestamps_token - 1;
                //Safe to unwrap, as we know the slice is never empty
                let &e_timestamp_token = tokens.last().unwrap();

                if e_timestamp_token == self.eot_token {
                    if s_timestamp == 0 || final_chunk {
                        if slice_len == m::N_SAMPLES || final_chunk {
                            self.buf.drain(..slice_len);
                            debug!(
                                num_tokens_decoded = dr.tokens.len(),
                                num_tokens_retained = 0,
                                "Transcribed all remaining data"
                            );
                        } else {
                            debug!(
                                num_tokens_decoded = dr.tokens.len(),
                                num_tokens_retained = tokens.len(),
                                "Transcribed, wating for more data"
                            );
                            break 'new_chunk;
                        };
                    } else {
                        let pre_drain_len = self.buf.len();
                        self.buf
                            .drain(..(s_timestamp as usize * 320).min(slice_len));

                        if pre_drain_len > slice_len {
                            debug!(
                                num_tokens_decoded = dr.tokens.len(),
                                num_tokens_retained = tokens.len(),
                                "Transcribed, getting a new slice"
                            );
                            break;
                        }

                        debug!(
                            num_tokens_decoded = dr.tokens.len(),
                            num_tokens_retained = tokens.len(),
                            "Transcribed, wating for more data"
                        );
                        break 'new_chunk;
                    };
                };

                if let Ok(text) = self.tokenizer.decode(&tokens[1..tokens.len() - 1], true) {
                    res.push_str(&text);
                };
            }
        }

        if final_chunk {
            self.lang.clear();
            self.model.reset_kv_cache();
        };

        Ok(res)
    }
}

impl Model {
    #[instrument(level = Level::DEBUG, skip_all)]
    fn decode_with_fallback(
        &mut self,
        mel: &Tensor,
    ) -> Result<Option<DecodingResult>, candle_core::Error> {
        let audio_features = self.model.encoder_forward(mel, true)?;

        if self.lang.is_none() {
            let lang = self.detect_language(&audio_features)?;
            self.lang.set_language_token(lang);
        }

        for &t in m::TEMPERATURES.iter() {
            let dr = self.decode(&audio_features, t)?;
            let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                || dr.avg_logprob < m::LOGPROB_THRESHOLD;
            if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                trace!(
                    at_temp = t,
                    logprob = dr.avg_logprob,
                    no_speech_prob = dr.no_speech_prob,
                    "Decoded with metrics"
                );
                return Ok(Some(dr));
            }
        }
        debug!("Failed to decode with all temps, returning None");
        Ok(None)
    }

    /// Panics if [`language_tokens_tensor`] is None
    fn detect_language(&mut self, audio_features: &Tensor) -> Result<u32, candle_core::Error> {
        let tokens = Tensor::new(&[[self.sot_token]], &self.device)?;
        let ys = self.model.decoder_forward(&tokens, audio_features, true)?;
        let logits = self.model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
        let logits = logits.index_select(self.lang.language_tokens_tensor().unwrap(), 0)?;
        let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;
        let probs = probs.to_vec1::<f32>()?;
        // Though this seems kinda strange, this should be faster then calling to_vec1() on
        // self.language_tokens_tensor.
        // TODO Verify
        let mut probs = Language::iter().zip(probs.iter()).collect::<Vec<_>>();
        probs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
        // Since the only possible outputs are from [`language_tokens_tensor`], so we can unwrap
        let language = self.tokenizer.token_to_id(probs[0].0.token()).unwrap();
        trace!(language = %probs[0].0, prob = probs[0].1, "Detected the language");
        Ok(language)
    }

    fn supress_timestamps(&self, logits: &Tensor) -> Result<Tensor, candle_core::Error> {
        logits.broadcast_add(&self.supress_timestamps)
    }

    fn supress_non_timestamps(
        &self,
        logits: &Tensor,
        last_timestep: u32,
    ) -> Result<Tensor, candle_core::Error> {
        let logits = self.supress_past_timestamps(logits, last_timestep)?;
        logits.broadcast_add(&self.supress_non_timestamps)
    }

    fn supress_past_timestamps(
        &self,
        logits: &Tensor,
        last_timestep: u32,
    ) -> Result<Tensor, candle_core::Error> {
        let len = logits.dim(D::Minus1)?;
        let supress_vec: Vec<f32> = (0..len)
            .map(|i| {
                if i > self.no_timestamps_token as usize && i <= last_timestep as usize {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let supress = Tensor::new(supress_vec.as_slice(), logits.device())?;
        let logits = logits.broadcast_add(&supress)?;
        Ok(logits)
    }

    fn supress_tokens(
        &self,
        logits: &Tensor,
        tokens: &[u32],
        last_timestep: u32,
    ) -> Result<Tensor, candle_core::Error> {
        let logits = logits.broadcast_add(&self.suppress_tokens)?;

        let l_token = tokens.last().unwrap();
        let sl_token = tokens.iter().nth_back(1);

        if l_token > &self.no_timestamps_token {
            if sl_token.is_some_and(|&token| token >= self.eot_token) {
                return self.supress_timestamps(&logits);
            }
            return self.supress_non_timestamps(&logits, last_timestep);
        }

        let sum_prob_timestamp = logits
            .i(self.no_timestamps_token as usize + 1..)?
            .sum(D::Minus1)?
            .to_scalar::<f32>()?;
        let prob_non_timestamp = logits
            .i(..self.no_timestamps_token as usize)?
            .max(D::Minus1)?
            .to_scalar::<f32>()?;

        if sum_prob_timestamp >= prob_non_timestamp {
            self.supress_non_timestamps(&logits, last_timestep)
        } else {
            self.supress_past_timestamps(&logits, last_timestep)
        }
    }

    fn decode(
        &mut self,
        audio_features: &Tensor,
        t: f64,
    ) -> Result<DecodingResult, candle_core::Error> {
        let mut sum_logprob = 0f64;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.lang.language_token() {
            tokens.push(language_token.to_owned());
        }
        tokens.push(self.task_token);

        let mut last_timestamp = None;

        let no_speech_prob = {
            let tokens_t = Tensor::new(tokens.as_slice(), audio_features.device())?;
            let tokens_t = tokens_t.unsqueeze(0)?;

            let ys = self
                .model
                .decoder_forward(&tokens_t, audio_features, true)?;
            let logits = self.model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;

            softmax(&logits, 0)?
                .i(self.no_speech_token as usize)?
                .to_scalar::<f32>()? as f64
        };

        while tokens.last().unwrap() != &self.eot_token {
            let tokens_t = Tensor::new(tokens.as_slice(), audio_features.device())?;
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = self
                .model
                .decoder_forward(&tokens_t, audio_features, false)?;

            let (_, seq_len, _) = ys.dims3()?;
            let logits = self
                .model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;

            let logits = softmax(&logits, D::Minus1)?;

            let logits = if let Some(lts) = last_timestamp {
                self.supress_tokens(&logits, &tokens, lts)?
            } else {
                // If this is the first token, force it to be a timestamp, in the range: [0..1]
                logits.broadcast_add(&self.first_token_supress)?
            };

            warn!(%logits, "logits");

            debug_assert!(!&logits
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .any(|&x| x != f32::NEG_INFINITY));

            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0).unwrap();
                warn!(%prs, "prs");
                debug_assert!(!&prs
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .any(|&x| x.is_nan()));
                let logits_v: Vec<f32> = prs.to_vec1()?;
                debug_assert!(!&logits_v.iter().any(|&x| x.is_nan()));
                let distr = rand::distributions::WeightedIndex::new(&logits_v).unwrap();
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };

            if next_token > self.no_timestamps_token {
                last_timestamp = Some(next_token);
            }

            tokens.push(next_token);
            let prob = logits.i(next_token as usize)?.to_scalar::<f32>()? as f64;
            sum_logprob += prob.ln();

            if tokens.len() >= self.model.config().max_target_positions - 1 {
                tokens.push(self.eot_token);
                break;
            }
        }

        let avg_logprob = sum_logprob / tokens.len() as f64;

        while tokens
            .iter()
            .nth_back(1)
            .is_some_and(|&t| t > self.no_timestamps_token)
        {
            tokens.remove(tokens.len() - 2);
        }

        Ok(DecodingResult {
            tokens,
            avg_logprob,
            no_speech_prob,
            compression_ratio: f64::NAN,
        })
    }
}

#[derive(Debug, strum::EnumIs)]
pub enum LanguageState {
    Detect {
        language_token: Option<u32>,
        language_tokens_tensor: Tensor,
    },
    // The id for the selected language
    ConstLang(u32),
}

impl LanguageState {
    pub fn is_none(&self) -> bool {
        match self {
            LanguageState::Detect { language_token, .. } => language_token.is_none(),
            LanguageState::ConstLang(_) => false,
        }
    }

    pub fn clear(&mut self) {
        match self {
            LanguageState::Detect { language_token, .. } => *language_token = None,
            LanguageState::ConstLang(_) => (),
        };
    }

    pub fn set_language_token(&mut self, lang: u32) {
        match self {
            LanguageState::Detect { language_token, .. } => *language_token = Some(lang),
            LanguageState::ConstLang(_) => (),
        };
    }

    pub fn language_tokens_tensor(&self) -> Option<&Tensor> {
        match self {
            LanguageState::Detect {
                language_tokens_tensor,
                ..
            } => Some(language_tokens_tensor),
            LanguageState::ConstLang(_) => None,
        }
    }

    pub fn language_token(&self) -> Option<&u32> {
        match self {
            LanguageState::Detect { language_token, .. } => language_token.as_ref(),
            LanguageState::ConstLang(t) => Some(t),
        }
    }
}

pub enum Type {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

impl Type {
    pub fn config(&self) -> &Config {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    pub fn encoder_forward(
        &mut self,
        x: &Tensor,
        flush: bool,
    ) -> Result<Tensor, candle_core::Error> {
        match self {
            Self::Normal(m) => m.encoder.forward(x, flush),
            Self::Quantized(m) => m.encoder.forward(x, flush),
        }
    }

    pub fn decoder_forward(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        flush: bool,
    ) -> Result<Tensor, candle_core::Error> {
        match self {
            Self::Normal(m) => m.decoder.forward(x, xa, flush),
            Self::Quantized(m) => m.decoder.forward(x, xa, flush),
        }
    }

    pub fn decoder_final_linear(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        match self {
            Self::Normal(m) => m.decoder.final_linear(x),
            Self::Quantized(m) => m.decoder.final_linear(x),
        }
    }

    pub fn reset_kv_cache(&mut self) {
        match self {
            Type::Normal(m) => m.reset_kv_cache(),
            Type::Quantized(m) => m.reset_kv_cache(),
        }
    }
}

#[derive(Debug, Clone)]
struct DecodingResult {
    tokens: Vec<u32>,
    avg_logprob: f64,
    no_speech_prob: f64,
    compression_ratio: f64,
}
