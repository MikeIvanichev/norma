use std::convert::Infallible;

use serde::{Deserialize, Serialize};

use super::{Model, ModelDefinition};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DummyDef {}

impl ModelDefinition for DummyDef {
    type Model = Dummy;

    fn common_params(&self) -> super::CommonModelParams {
        super::CommonModelParams {
            max_sample_len: 80000,
            data_buffer: 3,
            string_buffer: 3,
        }
    }
}

impl TryInto<Dummy> for DummyDef {
    type Error = Infallible;

    fn try_into(self) -> Result<Dummy, Self::Error> {
        Ok(Dummy {})
    }
}

pub struct Dummy {}

impl Model for Dummy {
    type Data = f32;

    const SAMPLE_RATE: u32 = 16_000;

    fn transcribe(&mut self, data: &mut [Self::Data]) -> String {
        format!("The len of this sample is: {}", data.len())
    }

    fn clear_context(&mut self) {}
}
