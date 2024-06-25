use dasp_sample::Duplex;

macro_rules! dtype {
    ($([$dtype:expr, $sample:expr] => $t:ty),+, _ => [$($it:ty),* $(,)?]  $(,)?) => {

        pub trait DType: cpal::SizedSample + dasp_frame::Frame + Send + Sync $(+Duplex<$it>)+ $(+Duplex<$t>)+ + 'static{
            fn to_dtype() -> candle_core::DType;
            fn to_sample_fromat() -> cpal::SampleFormat;
        }

        $(
        impl crate::dtype::DType for $t {
            fn to_dtype() -> candle_core::DType {
                $dtype
            }

            fn to_sample_fromat() -> cpal::SampleFormat {
                $sample
            }
        }
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
