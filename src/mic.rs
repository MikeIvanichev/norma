use serde::{Deserialize, Serialize};

#[derive(PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Settings {
    pub selected_device: Option<String>,
    pub on_error: OnError,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub enum OnError {
    Error,
    #[default]
    TryDefault,
}
