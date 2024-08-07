use std::fmt;

use strum::EnumIter;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, EnumIter)]
pub enum Language {
    English,
    Chinese,
    German,
    Spanish,
    Russian,
    Korean,
    French,
    Japanese,
    Portuguese,
    Turkish,
    Polish,
    Catalan,
    Dutch,
    Arabic,
    Swedish,
    Italian,
    Indonesian,
    Hindi,
    Finnish,
    Vietnamese,
    Hebrew,
    Ukrainian,
    Greek,
    Malay,
    Czech,
    Romanian,
    Danish,
    Hungarian,
    Tamil,
    Norwegian,
    Thai,
    Urdu,
    Croatian,
    Bulgarian,
    Lithuanian,
    Latin,
    Maori,
    Malayalam,
    Welsh,
    Slovak,
    Telugu,
    Persian,
    Latvian,
    Bengali,
    Serbian,
    Azerbaijani,
    Slovenian,
    Kannada,
    Estonian,
    Macedonian,
    Breton,
    Basque,
    Icelandic,
    Armenian,
    Nepali,
    Mongolian,
    Bosnian,
    Kazakh,
    Albanian,
    Swahili,
    Galician,
    Marathi,
    Punjabi,
    Sinhala,
    Khmer,
    Shona,
    Yoruba,
    Somali,
    Afrikaans,
    Occitan,
    Georgian,
    Belarusian,
    Tajik,
    Sindhi,
    Gujarati,
    Amharic,
    Yiddish,
    Lao,
    Uzbek,
    Faroese,
    HaitianCreole,
    Pashto,
    Turkmen,
    Nynorsk,
    Maltese,
    Sanskrit,
    Luxembourgish,
    Myanmar,
    Tibetan,
    Tagalog,
    Malagasy,
    Assamese,
    Tatar,
    Hawaiian,
    Lingala,
    Hausa,
    Bashkir,
    Javanese,
    Sundanese,
}

impl Language {
    /// Returns the token that represents a given language.
    ///
    /// ```
    /// use norma::models::whisper::Language;
    ///
    /// let token = Language::English.token();
    ///
    /// assert_eq!(token, "<|en|>");
    /// ```
    pub fn token(&self) -> &str {
        match self {
            Language::English => "<|en|>",
            Language::Chinese => "<|zh|>",
            Language::German => "<|de|>",
            Language::Spanish => "<|es|>",
            Language::Russian => "<|ru|>",
            Language::Korean => "<|ko|>",
            Language::French => "<|fr|>",
            Language::Japanese => "<|ja|>",
            Language::Portuguese => "<|pt|>",
            Language::Turkish => "<|tr|>",
            Language::Polish => "<|pl|>",
            Language::Catalan => "<|ca|>",
            Language::Dutch => "<|nl|>",
            Language::Arabic => "<|ar|>",
            Language::Swedish => "<|sv|>",
            Language::Italian => "<|it|>",
            Language::Indonesian => "<|id|>",
            Language::Hindi => "<|hi|>",
            Language::Finnish => "<|fi|>",
            Language::Vietnamese => "<|vi|>",
            Language::Hebrew => "<|he|>",
            Language::Ukrainian => "<|uk|>",
            Language::Greek => "<|el|>",
            Language::Malay => "<|ms|>",
            Language::Czech => "<|cs|>",
            Language::Romanian => "<|ro|>",
            Language::Danish => "<|da|>",
            Language::Hungarian => "<|hu|>",
            Language::Tamil => "<|ta|>",
            Language::Norwegian => "<|no|>",
            Language::Thai => "<|th|>",
            Language::Urdu => "<|ur|>",
            Language::Croatian => "<|hr|>",
            Language::Bulgarian => "<|bg|>",
            Language::Lithuanian => "<|lt|>",
            Language::Latin => "<|la|>",
            Language::Maori => "<|mi|>",
            Language::Malayalam => "<|ml|>",
            Language::Welsh => "<|cy|>",
            Language::Slovak => "<|sk|>",
            Language::Telugu => "<|te|>",
            Language::Persian => "<|fa|>",
            Language::Latvian => "<|lv|>",
            Language::Bengali => "<|bn|>",
            Language::Serbian => "<|sr|>",
            Language::Azerbaijani => "<|az|>",
            Language::Slovenian => "<|sl|>",
            Language::Kannada => "<|kn|>",
            Language::Estonian => "<|et|>",
            Language::Macedonian => "<|mk|>",
            Language::Breton => "<|br|>",
            Language::Basque => "<|eu|>",
            Language::Icelandic => "<|is|>",
            Language::Armenian => "<|hy|>",
            Language::Nepali => "<|ne|>",
            Language::Mongolian => "<|mn|>",
            Language::Bosnian => "<|bs|>",
            Language::Kazakh => "<|kk|>",
            Language::Albanian => "<|sq|>",
            Language::Swahili => "<|sw|>",
            Language::Galician => "<|gl|>",
            Language::Marathi => "<|mr|>",
            Language::Punjabi => "<|pa|>",
            Language::Sinhala => "<|si|>",
            Language::Khmer => "<|km|>",
            Language::Shona => "<|sn|>",
            Language::Yoruba => "<|yo|>",
            Language::Somali => "<|so|>",
            Language::Afrikaans => "<|af|>",
            Language::Occitan => "<|oc|>",
            Language::Georgian => "<|ka|>",
            Language::Belarusian => "<|be|>",
            Language::Tajik => "<|tg|>",
            Language::Sindhi => "<|sd|>",
            Language::Gujarati => "<|gu|>",
            Language::Amharic => "<|am|>",
            Language::Yiddish => "<|yi|>",
            Language::Lao => "<|lo|>",
            Language::Uzbek => "<|uz|>",
            Language::Faroese => "<|fo|>",
            Language::HaitianCreole => "<|ht|>",
            Language::Pashto => "<|ps|>",
            Language::Turkmen => "<|tk|>",
            Language::Nynorsk => "<|nn|>",
            Language::Maltese => "<|mt|>",
            Language::Sanskrit => "<|sa|>",
            Language::Luxembourgish => "<|lb|>",
            Language::Myanmar => "<|my|>",
            Language::Tibetan => "<|bo|>",
            Language::Tagalog => "<|tl|>",
            Language::Malagasy => "<|mg|>",
            Language::Assamese => "<|as|>",
            Language::Tatar => "<|tt|>",
            Language::Hawaiian => "<|haw|>",
            Language::Lingala => "<|ln|>",
            Language::Hausa => "<|ha|>",
            Language::Bashkir => "<|ba|>",
            Language::Javanese => "<|jw|>",
            Language::Sundanese => "<|su|>",
        }
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let full_name = match self {
            Language::English => "English",
            Language::Chinese => "Chinese",
            Language::German => "German",
            Language::Spanish => "Spanish",
            Language::Russian => "Russian",
            Language::Korean => "Korean",
            Language::French => "French",
            Language::Japanese => "Japanese",
            Language::Portuguese => "Portuguese",
            Language::Turkish => "Turkish",
            Language::Polish => "Polish",
            Language::Catalan => "Catalan",
            Language::Dutch => "Dutch",
            Language::Arabic => "Arabic",
            Language::Swedish => "Swedish",
            Language::Italian => "Italian",
            Language::Indonesian => "Indonesian",
            Language::Hindi => "Hindi",
            Language::Finnish => "Finnish",
            Language::Vietnamese => "Vietnamese",
            Language::Hebrew => "Hebrew",
            Language::Ukrainian => "Ukrainian",
            Language::Greek => "Greek",
            Language::Malay => "Malay",
            Language::Czech => "Czech",
            Language::Romanian => "Romanian",
            Language::Danish => "Danish",
            Language::Hungarian => "Hungarian",
            Language::Tamil => "Tamil",
            Language::Norwegian => "Norwegian",
            Language::Thai => "Thai",
            Language::Urdu => "Urdu",
            Language::Croatian => "Croatian",
            Language::Bulgarian => "Bulgarian",
            Language::Lithuanian => "Lithuanian",
            Language::Latin => "Latin",
            Language::Maori => "Maori",
            Language::Malayalam => "Malayalam",
            Language::Welsh => "Welsh",
            Language::Slovak => "Slovak",
            Language::Telugu => "Telugu",
            Language::Persian => "Persian",
            Language::Latvian => "Latvian",
            Language::Bengali => "Bengali",
            Language::Serbian => "Serbian",
            Language::Azerbaijani => "Azerbaijani",
            Language::Slovenian => "Slovenian",
            Language::Kannada => "Kannada",
            Language::Estonian => "Estonian",
            Language::Macedonian => "Macedonian",
            Language::Breton => "Breton",
            Language::Basque => "Basque",
            Language::Icelandic => "Icelandic",
            Language::Armenian => "Armenian",
            Language::Nepali => "Nepali",
            Language::Mongolian => "Mongolian",
            Language::Bosnian => "Bosnian",
            Language::Kazakh => "Kazakh",
            Language::Albanian => "Albanian",
            Language::Swahili => "Swahili",
            Language::Galician => "Galician",
            Language::Marathi => "Marathi",
            Language::Punjabi => "Punjabi",
            Language::Sinhala => "Sinhala",
            Language::Khmer => "Khmer",
            Language::Shona => "Shona",
            Language::Yoruba => "Yoruba",
            Language::Somali => "Somali",
            Language::Afrikaans => "Afrikaans",
            Language::Occitan => "Occitan",
            Language::Georgian => "Georgian",
            Language::Belarusian => "Belarusian",
            Language::Tajik => "Tajik",
            Language::Sindhi => "Sindhi",
            Language::Gujarati => "Gujarati",
            Language::Amharic => "Amharic",
            Language::Yiddish => "Yiddish",
            Language::Lao => "Lao",
            Language::Uzbek => "Uzbek",
            Language::Faroese => "Faroese",
            Language::HaitianCreole => "Haitian Creole",
            Language::Pashto => "Pashto",
            Language::Turkmen => "Turkmen",
            Language::Nynorsk => "Nynorsk",
            Language::Maltese => "Maltese",
            Language::Sanskrit => "Sanskrit",
            Language::Luxembourgish => "Luxembourgish",
            Language::Myanmar => "Myanmar",
            Language::Tibetan => "Tibetan",
            Language::Tagalog => "Tagalog",
            Language::Malagasy => "Malagasy",
            Language::Assamese => "Assamese",
            Language::Tatar => "Tatar",
            Language::Hawaiian => "Hawaiian",
            Language::Lingala => "Lingala",
            Language::Hausa => "Hausa",
            Language::Bashkir => "Bashkir",
            Language::Javanese => "Javanese",
            Language::Sundanese => "Sundanese",
        };
        write!(f, "{full_name}")
    }
}
