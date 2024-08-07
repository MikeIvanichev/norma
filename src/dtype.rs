use dasp_sample::Duplex;
use std::fmt::Debug;

mod sealed {
    pub trait Sealed {}
}

use sealed::Sealed;

macro_rules! dtype {
    ($([$dtype:expr, $sample:expr] => $t:ty),+, _ => [$($it:ty),* $(,)?]  $(,)?) => {

        pub trait DType: Sealed + cpal::SizedSample + dasp_frame::Frame + Send + Sync + Debug $(+Duplex<$it>)+ $(+Duplex<$t>)+ + 'static{
            #[cfg(feature = "whisper")]
            fn to_dtype() -> candle_core::DType;

            fn to_sample_fromat() -> cpal::SampleFormat;
        }

        $(
        impl crate::dtype::DType for $t {
            #[cfg(feature = "whisper")]
            fn to_dtype() -> candle_core::DType {
                $dtype
            }

            fn to_sample_fromat() -> cpal::SampleFormat {
                $sample
            }
        }

        impl Sealed for $t {}
        )+
    };
}

dtype!(
    // These are valid DTypes
    [candle_core::DType::U8, cpal::SampleFormat::U8] => u8,
    [candle_core::DType::U32, cpal::SampleFormat::U32] => u32,
    [candle_core::DType::F32, cpal::SampleFormat::F32] => f32,
    [candle_core::DType::F64, cpal::SampleFormat::F64] => f64,
    // These are not valid DTypes, but can (will) be converted to a valid DType
    _ => [i8, i16, i32, i64, u16, u64],
);
