import json, os, sys, string, time

# Folder input/output
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(SCRIPT_DIR, "../stage/raw")
output_dir = os.path.join(SCRIPT_DIR, "../stage/interim/cleaned")
sys.stdout.reconfigure(
    encoding="utf-8", errors="replace"
)  # cegah UnicodeEncodeError di konsol Windows (cp1252)

# Buat folder output jika belum ada
os.makedirs(output_dir, exist_ok=True)


# Cetak progress per 1% pada baris yang sama
def print_progress(i, total):
    step = max(1, total // 100)
    if i % step == 0 or i == total:
        print(
            f"    Progress: {i / total * 100:.0f}% ({i}/{total})", end="\r", flush=True
        )


def clean_az(text):  # Fungsi pembersih teks

    # Ubah ke huruf kecil dan bentuk filter karakter
    lower_text = text.lower()
    allowed = string.ascii_lowercase + " "  # String a-z dan spasi
    result_text = []

    # Filter karakter menjadi huruf dan tidak kapital (a-z dan spasi saja).
    for char in lower_text:
        if char in allowed:
            result_text.append(char)

    # Gabungkan list of chars menjadi string lagi
    cleaned_text = "".join(result_text)

    # Rapikan spasi: buang spasi di tepi dan ubah spasi ganda menjadi satu
    return " ".join(cleaned_text.split())


start_all = time.perf_counter()
files = [f for f in os.listdir(input_dir) if f.endswith(".json")]  # File .JSON saja
print(f"Ditemukan {len(files)} file JSON di: {input_dir}")

for n, file in enumerate(files, start=1):  # Iterasi semua file di folder
    print(f"[{n}/{len(files)}] Membaca {file!r} ...")
    filepath = os.path.join(input_dir, file)
    try:
        # Baca file JSON
        with open(filepath, "r", encoding="utf-8") as input_file:
            data = json.load(input_file)
    except Exception as e:
        print(f"  Error membaca {file!r}: {e}")
        continue

    cleaned_sentences = []
    # Gabungkan SEMUA percakapan di file (sebelumnya hanya yang pertama)
    conversation = [line for conv in data.get("conversations", []) for line in conv]
    total = len(conversation)
    print(f"  Membersihkan {total} baris")

    # Ambil string kalimat berdasarkan bentuk dataset
    for i, line in enumerate(conversation, start=1):
        sentence = line[0]
        cleaned = clean_az(sentence)
        # Hanya simpan kalimat yang tidak kosong
        if cleaned:
            cleaned_sentences.append(cleaned)
        print_progress(i, total)
    print()

    # Buat data output dalam format JSON
    output_data = {"data": cleaned_sentences}

    # Tulis hasil ke file baru di folder output
    output_path = os.path.join(output_dir, file)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(output_data, output_file, indent=2)

    print(
        f"  Tersimpan: {len(cleaned_sentences)} kalimat "
        f"({total - len(cleaned_sentences)} kosong dibuang)"
    )

print(f"Selesai dalam {time.perf_counter() - start_all:.1f} detik.")
print("Hasil penyesuaian disimpan di:", output_dir)
