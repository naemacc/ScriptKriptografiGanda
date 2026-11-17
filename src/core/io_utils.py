import json, time
from pathlib import Path


def save_json(path: Path, data, indent=4):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_dataset(path: Path) -> tuple[int, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    return content["total"], content["data"]


# Buat folder berstempel waktu; tunggu jika folder dengan stempel sama sudah ada
def new_run_dir(base: Path) -> Path:
    while True:
        out_dir = base / time.strftime("%Y-%m-%d_%H-%M-%S")
        try:
            out_dir.mkdir(parents=True)
            return out_dir
        except FileExistsError:
            time.sleep(0.5)


def save_category(
    out_dir: Path,
    category: str,
    ciphertexts: list[str],
    times_s: list[float],
    entropy: list[float],
    avalanche: float,
    plaintexts: list[str] | None = None,
):
    # URUTAN DATASET DIPERTAHANKAN (tidak diurutkan berdasarkan waktu), sehingga
    # indeks ke-i di semua run/metode = kalimat yang sama.
    # Tiap entri mencatat plaintext, ciphertext, dan waktu enkripsinya (µs).
    if plaintexts is None:
        plaintexts = [None] * len(ciphertexts)

    timed = [
        {"plaintext": p, "ciphertext": c, "execTime": t * 1e6}  # detik -> mikrodetik
        for p, c, t in zip(plaintexts, ciphertexts, times_s)
    ]

    save_json(
        out_dir / f"{category}.json",
        {
            "category": category,
            "total": len(timed),
            "entropy": entropy,
            "avalanche": avalanche,
            "data": timed,
        },
        indent=None,  # kompak: file 1 juta kalimat jauh lebih kecil
    )


# Menambahkan entri ke file log, membuat file jika belum ada
def append_log(path: Path, entry):
    logs = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
    logs.append(entry)
    save_json(path, logs)
