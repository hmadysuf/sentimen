"""Modul preprocessing teks bahasa Indonesia."""
import re
import string
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi Sastrawi
_stemmer = StemmerFactory().create_stemmer()
_stopword = StopWordRemoverFactory().create_stop_word_remover()

# Kamus normalisasi kata alay/singkatan
NORMALIZATION_DICT = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "tdk": "tidak",
    "engga": "tidak", "enggak": "tidak", "nggak": "tidak", "ngga": "tidak",
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "krn": "karena",
    "karna": "karena", "sdh": "sudah", "udh": "sudah", "blm": "belum",
    "tp": "tapi", "tpi": "tapi", "bgt": "banget", "sm": "sama",
    "aja": "saja", "aj": "saja", "d": "di", "dr": "dari",
    "org": "orang", "pemerintah": "pemerintah", "mrk": "mereka",
    "kalo": "kalau", "kl": "kalau", "jd": "jadi", "jgn": "jangan",
    "bikin": "buat", "makanan": "makanan", "mkn": "makan",
    "anak2": "anak-anak", "anak²": "anak-anak",
    "mbg": "mbg", "makan bergizi gratis": "mbg",
}

# Kata negasi penting
NEGATION_WORDS = {"tidak", "bukan", "belum", "jangan", "tanpa", "kurang"}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # hapus URL
    text = re.sub(r"@\w+|#\w+", " ", text)                 # hapus mention/hashtag
    text = re.sub(r"\d+", " ", text)                       # hapus angka
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str) -> str:
    tokens = text.split()
    return " ".join(NORMALIZATION_DICT.get(t, t) for t in tokens)


def remove_stopwords(text: str) -> str:
    # Stopword Sastrawi, tapi pertahankan kata negasi
    tokens = text.split()
    result = []
    for t in tokens:
        if t in NEGATION_WORDS:
            result.append(t)
            continue
        if _stopword.remove(t).strip() == "":
            continue
        result.append(t)
    return " ".join(result)


def stem_text(text: str) -> str:
    return _stemmer.stem(text)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "comment_text") -> pd.DataFrame:
    """Pipeline preprocessing lengkap pada DataFrame."""
    df = df.copy()
    df = df.dropna(subset=[text_col])
    df = df.drop_duplicates(subset=[text_col])

    df["clean"] = df[text_col].apply(clean_text)
    df["normalized"] = df["clean"].apply(normalize_text)
    df["no_stopword"] = df["normalized"].apply(remove_stopwords)
    df["stemmed"] = df["no_stopword"].apply(stem_text)
    df["tokens"] = df["stemmed"].apply(lambda x: x.split())
    df = df[df["stemmed"].str.strip() != ""]
    return df.reset_index(drop=True)