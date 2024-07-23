use std::{thread, time::Duration};

use norma::{models::whisper, Transcriber};

fn main() {
    let model_definition = whisper::monolingual::Definition::new(
        whisper::monolingual::ModelType::DistilMediumEn,
        norma::models::SelectedDevice::Cpu,
    );

    let (jh, th) = Transcriber::blocking_spawn(model_definition).unwrap();

    let mut stream = th.blocking_start(norma::mic::Settings::default()).unwrap();

    thread::spawn(move || {
        while let Some(seg) = stream.blocking_recv() {
            println!("{}", seg);
        }
    });

    thread::sleep(Duration::from_secs_f32(10f32));
    th.stop().unwrap();
    drop(th);

    jh.join().unwrap().unwrap();
}
