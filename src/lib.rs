pub mod mic;
pub mod model;

use std::{
    sync::mpsc::{channel, Receiver, Sender},
    thread::JoinHandle,
    time::Duration,
};

use mic::MicSettings;
use model::ModelDefinition;
use serde::{Deserialize, Serialize};

pub struct Norma<T>
where
    T: ModelDefinition,
{
    model_definition: T,
    mic_settings: MicSettings,
    mode: Mode,
    _jh: Option<JoinHandle<usize>>,
}

impl<T> Norma<T>
where
    T: ModelDefinition,
{
    fn start_transcription() -> Receiver<String> {
        let (tx, rx) = channel();

        rx
    }

    fn end_transcription() {}

    fn set_mode(&mut self, mode: Mode) {
        if mode == self.mode {
            return;
        }

        match mode {
            Mode::Oneshot => todo!(),
            Mode::Presist(_) => todo!(),
        }
    }
}

#[derive(Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Mode {
    #[default]
    Oneshot,
    Presist(Duration),
}

mod private {
    pub trait Sealed {}
    pub struct Token;
}
