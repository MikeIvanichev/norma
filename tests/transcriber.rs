use std::{thread::sleep, time::Duration};

use norma::{models::mock::MockDef, Transcriber, TranscriberHandle};

#[test]
fn blocking_mock_model() {
    let (jh, th) = Transcriber::blocking_spawn(MockDef {}).unwrap();

    let mut stream = th.blocking_start(norma::mic::Settings::default()).unwrap();
    sleep(Duration::from_secs_f64(1f64));
    th.stop().unwrap();

    assert_eq!(stream.try_recv(), Ok(String::new()));

    jh.join().unwrap().unwrap();
}

#[tokio::test]
async fn mock_model() {
    let (jh, th) = Transcriber::spawn(MockDef {}).await.unwrap();

    let mut stream = th.start(norma::mic::Settings::default()).await.unwrap();
    tokio::time::sleep(Duration::from_secs_f64(1f64)).await;
    th.stop().unwrap();

    assert_eq!(stream.try_recv(), Ok(String::new()));

    jh.join().unwrap().unwrap();
}
