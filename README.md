# Norma

An easy to use and extensible
pure Rust real time tranascription (speach to text) library.

[![Latest version](https://img.shields.io/crates/v/norma.svg)](https://crates.io/crates/norma)
[![Documentation](https://docs.rs/norma/badge.svg)](https://docs.rs/norma)
![License](https://img.shields.io/crates/l/norma.svg)

## Audio backends

Norma uses [cpal](https://github.com/RustAudio/cpal)
so as to be agnostice over mutiple audio backends.

This allows us to suppoert:

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

## Acceleraters

An accelerater can be selected using ['norma::SelectedDevice'].

### CPU and Accelerate

Using the CPU does not require any extra features, however
when building on MacOS the `accelerate` feature can be enabled to allow
the resulting program to utilize Apples [Accelerate framwork](https://developer.apple.com/accelerate/).

```rust
let device = SelectedDevice::Cpu;
```

### CUDA and cuDNN

In order for the below code to compile either the `cuda`
or the `cudnn` feature must be enabled.

The `cuda` feature flag requires that CUDA
be installed and correctly configured on your machine.
One enabled the program will be built with CUDA support,
and require CUDA be installed on the machine running the code.

The `cudnn` feature flag requires that cuDNN
be installed and correctly configured on your machine.
One enabled the program will be built with cuDNN support,
and require CUDA and cuDNN be installed on the machine running the code.

```rust
let device = SelectedDevice::Cuda(ord);
```

Where `ord` is the ID of the CUDA device you want to use,
if you only have one device or want to use the default set it to 0.

### Metal

Using the Metal accelerater requires compiling the program
with the `metal` feature flag.

```rust
let device = SelectedDevice::Metal;
```

## Models

- Whisper (with full long form decoding support)
