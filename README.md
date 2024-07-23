# Norma

An easy to use and extensible
pure Rust real-time transcription (speech-to-text) library.

[![Latest version](https://img.shields.io/crates/v/norma.svg)](https://crates.io/crates/norma)
[![Documentation](https://docs.rs/norma/badge.svg)](https://docs.rs/norma)
![License](https://img.shields.io/crates/l/norma.svg)

## Models

- Whisper (with full long-form decoding support)

## Exmaple

```rust
use std::{
    thread::{self, sleep},
    time::Duration,
};
use norma::{
    mic::Settings,
    models::whisper::monolingual,
    Transcriber,
};

// Define the model that will be used for transcription
let model = monolingual::Definition::new(
    monolingual::ModelType::DistilLargeEnV3,
    norma::models::SelectedDevice::Cpu, // Replace with Cuda(0) or Metal as needed
);

// Spawn the transcriber in a new std thread
let (jh, th) = Transcriber::blocking_spawn(model).unwrap();

// Start recording using the default microphone
let mut stream = th.blocking_start(Settings::default()).unwrap();

thread::spawn(move || while let Some(msg) = stream.blocking_recv() {
  println!("{}", msg);
});

sleep(Duration::from_secs(10));

// Stop the transcription and drop the TranscriberHandle,
// causing the transcriber to terminate
th.stop().unwrap();
drop(th);

// Join the thread that was spawned for the transcriber
jh.join().unwrap().unwrap();
```

## Audio backends

Norma uses [cpal](https://github.com/RustAudio/cpal)
to be agnostic over multiple audio backends.

This allows us to support:

- Linux (via ALSA or JACK)
- Windows (via WASAPI)
- macOS (via CoreAudio)
- iOS (via CoreAudio)
- Android (via Oboe)

Some audio backends are optional and will only be compiled with a feature flag.

- JACK (on Linux): `jack`

Oboe can either use a shared or static runtime.
The static runtime is used by default,
but activating the `oboe-shared-stdcxx` feature makes it use the shared runtime,
which requires libc++\_shared.so from the Android NDK to be present during execution.

## Accelerators

All Accelerators are defined in `norma::SelectedDevice`.

### CPU

Using the CPU does not require any extra features.

However when building on MacOS the `accelerate` feature can be enabled to allow
the resulting program to utilize Apple's [Accelerate framwork](https://developer.apple.com/accelerate/).

```rust
let device = SelectedDevice::Cpu;
```

### CUDA and cuDNN

For the below code to compile either the `cuda`
or the `cudnn` feature must be enabled.

The `cuda` feature flag requires that CUDA
be installed and correctly configured on your machine.
Once enabled the program will be built with CUDA support,
and require CUDA on the machine running the code.

The `cudnn` feature flag requires that cuDNN
be installed and correctly configured on your machine.
Once enabled the program will be built with cuDNN support,
and require CUDA and cuDNN on the machine running the code.

```rust
let device = SelectedDevice::Cuda(ord);
```

Where `ord` is the ID of the CUDA device you want to use.
If you only have one device or want to use the default set it to 0.

### Metal

Using the Metal requires compiling the program on MacOS
with the `metal` feature flag.

```rust
let device = SelectedDevice::Metal;
```
