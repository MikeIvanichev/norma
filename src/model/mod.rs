pub mod whisper;

/// The model should be able to provide a data and error handler and then spawn itself in another
/// thread.
///
/// It is expected
pub trait ModelDefinition {
    type Data;

    fn data_handler<D, T>(&self, tx: std::sync::mpsc::Sender<Self::Data>) -> D
    where
        D: FnMut(&[T], &cpal::InputCallbackInfo) + Send + 'static;

    fn error_handler<E>(&self) -> E
    where
        E: FnMut(cpal::StreamError) + Send + 'static;

    fn run(
        &self,
        rx: std::sync::mpsc::Receiver<Self::Data>,
    ) -> (
        std::sync::mpsc::Receiver<String>,
        std::thread::JoinHandle<usize>,
    );
}
