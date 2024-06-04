use serde::{Deserialize, Serialize};

#[derive(PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct MicSettings {
    on_error: OnError,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub enum OnError {
    Error,
    #[default]
    TryDefault,
}
