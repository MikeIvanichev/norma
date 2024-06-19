use serde::{de::DeserializeOwned, Serialize};
use std::error::Error;

use crate::{dtype::DType, private::Sealed};

pub mod whisper;

pub enum ModelInput<T> {
    Data(T),
    ClearContext,
}

pub trait ModelDefinition: Serialize + DeserializeOwned + PartialEq + Sealed {
    const SAMPLE_RATE: u32;
    const MAX_SAMPLE_LEN: usize;
    type Data: DType;
    type TranscriptionError: Error;

    fn run(
        &self,
    ) -> Result<
        impl FnMut(ModelInput<&mut [Self::Data]>) -> Result<String, Self::TranscriptionError>
            + Send
            + 'static,
        impl Error,
    >;
}
