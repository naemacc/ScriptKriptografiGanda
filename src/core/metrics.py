import math, random
from collections import Counter

from core.ciphers import ALPHABET

AVALANCHE_SAMPLE = 1000  # jumlah kalimat yang diuji
AVALANCHE_SEED = 42  # seed tetap -> sampel sama untuk kedua metode


def entropy(text: str) -> float:
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def avalanche(plaintexts: list[str], encrypt) -> float:
    """Rata-rata persentase karakter ciphertext yang berubah ketika SATU
    karakter plaintext diubah (karakter diganti dengan karakter berikutnya
    di ALPHABET). Posisi yang diubah dipilih acak (seed tetap) pada
    sampel kalimat yang sama untuk setiap metode."""
    rng = random.Random(AVALANCHE_SEED)
    candidates = [p for p in plaintexts if p]
    sample = rng.sample(candidates, min(AVALANCHE_SAMPLE, len(candidates)))

    rates = []
    for plain in sample:
        pos = rng.randrange(len(plain))
        idx = ALPHABET.index(plain[pos])
        modified = plain[:pos] + ALPHABET[(idx + 1) % 27] + plain[pos + 1 :]

        c1, c2 = encrypt(plain), encrypt(modified)
        diff = sum(a != b for a, b in zip(c1, c2)) + abs(len(c1) - len(c2))
        rates.append(diff / max(len(c1), len(c2)) * 100)
    return sum(rates) / len(rates) if rates else 0.0


def validate(plaintexts: list[str], ciphertexts: list[str], encrypt):
    """Returns ([entropy_plain, entropy_cipher], avalanche_pct)."""
    ent = [entropy("".join(plaintexts)), entropy("".join(ciphertexts))]
    return ent, avalanche(plaintexts, encrypt)
