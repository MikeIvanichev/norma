use std::{thread::sleep, time::Duration};

use norma::{
    models::mock::{MockDef, FINAL_MSG, MSG},
    Transcriber,
};

#[test]
#[ignore = "Requires a mic to run"]
fn blocking_mock_model() {
    let (jh, th) = Transcriber::blocking_spawn(MockDef {}).unwrap();

    let mut stream = th.blocking_start(norma::mic::Settings::default()).unwrap();
    sleep(Duration::from_secs_f64(3f64));
    th.stop().unwrap();
    drop(th);

    let mut res = Vec::new();

    while let Some(msg) = stream.blocking_recv() {
        res.push(msg);
    }

    assert!(!res.is_empty(), "Expected non-empty message list");

    for msg in &res {
        assert!(*msg == MSG || *msg == FINAL_MSG, "Unexpected message type");
    }

    assert_eq!(
        res.iter().filter(|msg| *msg == FINAL_MSG).count(),
        1,
        "Expected exactly one FINAL_MSG"
    );

    jh.join().unwrap().unwrap();
}

#[tokio::test]
#[ignore = "Requires a mic to run"]
async fn mock_model() {
    let (jh, th) = Transcriber::spawn(MockDef {}).await.unwrap();

    let mut stream = th.start(norma::mic::Settings::default()).await.unwrap();
    tokio::time::sleep(Duration::from_secs_f64(3f64)).await;
    th.stop().unwrap();
    drop(th);

    let mut res = Vec::new();

    while let Some(msg) = stream.recv().await {
        res.push(msg);
    }

    assert!(!res.is_empty(), "Expected non-empty message list");

    for msg in &res {
        assert!(*msg == MSG || *msg == FINAL_MSG, "Unexpected message type");
    }

    assert_eq!(
        res.iter().filter(|msg| *msg == FINAL_MSG).count(),
        1,
        "Expected exactly one FINAL_MSG"
    );

    jh.join().unwrap().unwrap();
}
