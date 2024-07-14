use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Settings {
    pub selected_device: Option<String>,
    pub on_error: OnError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum OnError {
    Error,
    #[default]
    TryDefault,
}
