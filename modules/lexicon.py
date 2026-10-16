"""Leksikon sentimen sederhana bahasa Indonesia (dapat diperluas)."""

POSITIVE_WORDS = {
    "bagus", "baik", "bermanfaat", "membantu", "setuju", "dukung", "mendukung",
    "mantap", "keren", "hebat", "luar", "biasa", "sukses", "berhasil", "maju",
    "senang", "bahagia", "terima", "kasih", "apresiasi", "positif", "setuju",
    "layak", "tepat", "cerdas", "peduli", "sejahtera", "makmur", "sehat",
    "bergizi", "enak", "murah", "gratis", "mulia", "terbaik", "solusi",
    "program", "bagus", "lanjut", "semangat", "semoga", "sukses",
}

NEGATIVE_WORDS = {
    "buruk", "jelek", "gagal", "tolak", "menolak", "tidak setuju", "korupsi",
    "rugi", "bohong", "dusta", "tipu", "penipuan", "salah", "kacau", "parah",
    "kecewa", "marah", "benci", "sedih", "khawatir", "takut", "bahaya",
    "mahal", "mubazir", "buang", "sia-sia", "omong", "kosong", "pencitraan",
    "politis", "kampanye", "hentikan", "stop", "batal", "gagal", "susah",
    "sulit", "masalah", "beban", "utang", "korup", "nepotisme",
}

NEGATION_WORDS = {"tidak", "bukan", "belum", "jangan", "tanpa", "kurang"}


def score_text(tokens: list) -> int:
    """Menghitung skor sentimen dengan mempertimbangkan negasi."""
    score = 0
    for i, tok in enumerate(tokens):
        negated = i > 0 and tokens[i - 1] in NEGATION_WORDS
        if tok in POSITIVE_WORDS:
            score += -1 if negated else 1
        elif tok in NEGATIVE_WORDS:
            score += 1 if negated else -1
    return score


def label_from_score(score: int) -> str:
    if score > 0:
        return "positif"
    elif score < 0:
        return "negatif"
    return "netral"