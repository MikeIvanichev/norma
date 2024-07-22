/// This is intended solely for testing, this model will allways returb an empty string.
use std::convert::Infallible;

use super::{Model, ModelDefinition};

const SAMPLE_RATE: u32 = 44_100;
pub const MSG: &str = "Mock Model";
pub const FINAL_MSG: &str = "Mock Model Out";

#[derive(Debug, Clone, Copy)]
pub struct MockDef {}

impl ModelDefinition for MockDef {
    type Model = Mock;

    type Error = Infallible;

    fn common_params(&self) -> &super::CommonModelParams {
        &super::CommonModelParams {
            max_chunk_len: SAMPLE_RATE as usize,
            data_buffer_size: 3,
            string_buffer_size: 3,
        }
    }

    async fn try_to_model(self) -> Result<Self::Model, Self::Error> {
        Ok(Mock {})
    }

    fn blocking_try_to_model(self) -> Result<Self::Model, Self::Error> {
        Ok(Mock {})
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Mock {}

impl Model for Mock {
    type Data = f64;

    type Error = Infallible;

    // Standard windows sample rate
    const SAMPLE_RATE: u32 = SAMPLE_RATE;

    fn transcribe(
        &mut self,
        _data: &mut Vec<Self::Data>,
        final_chunk: bool,
    ) -> Result<String, Self::Error> {
        if !final_chunk {
            Ok(MSG.to_string())
        } else {
            Ok(FINAL_MSG.to_string())
        }
    }
}
