import json, os, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from config import RUNS_DIR

# Variabel global dan konfigurasi
sns.set_theme(style="whitegrid")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 42
MAX_BOX_POINTS = (
    50_000  # subsampel per grup HANYA untuk boxplot (uji-t memakai semua data)
)
METHODS = ["Cross", "Sequenced"]
PALETTE = dict(zip(METHODS, sns.color_palette()[:2]))

# Hasil analisis (grafik dan log) disimpan di results/, terpisah dari kode
RESULTS_DIR = RUNS_DIR.parent
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Folder input: results/runs/<skenario>/<metode>/
input_dir_scale_group1 = RUNS_DIR / "scale" / "cross"
input_dir_scale_group2 = RUNS_DIR / "scale" / "sequenced"
input_dir_sense_group1 = RUNS_DIR / "sense" / "cross"
input_dir_sense_group2 = RUNS_DIR / "sense" / "sequenced"


def simpan_grafik(nama):
    plt.savefig(FIGURES_DIR / nama, dpi=300)
    plt.close()


def save_log(label: str, data):
    path = os.path.join(RESULTS_DIR, f"{label}_log.json")

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            try:
                logs = json.load(file)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    logs.append(data)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(logs, file, ensure_ascii=False, indent=4)


def format_n(n):
    # 100000 -> "100k", 1000000 -> "1M"
    return f"{n / 1e6:g}M" if n >= 1e6 else f"{n / 1e3:g}k"


def load_data(base_dir, method_name, is_sensitivity=False):
    # Mengembalikan (df_summary, times) dengan times = {kategori: [array waktu (µs) per run, urutan dataset asli]}
    summary_data = []
    times = {}

    if is_sensitivity:
        mapping = {"pendek": "Pendek", "menengah": "Menengah", "panjang": "Panjang"}
    else:
        mapping = {"sedikit": "Sedikit", "sedang": "Sedang", "banyak": "Banyak"}

    if not os.path.exists(base_dir):
        print(f"[Peringatan] Direktori tidak ditemukan: {base_dir}")
        save_log("hasil", {"Peringatan": f"Direktori tidak ditemukan: {base_dir}"})
        return pd.DataFrame(), {}

    # Iterasi melalui semua folder run (nama folder berupa timestamp)
    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for cat_key, cat_label in mapping.items():
            file_path = os.path.join(folder_path, f"{cat_key}.json")
            if not os.path.exists(file_path):
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = data["data"]
            exec_times = np.fromiter(
                (item["execTime"] for item in items), dtype=float, count=len(items)
            )
            total_time = float(exec_times.sum())
            mean_time = float(exec_times.mean()) if exec_times.size else 0.0

            summary_data.append(
                {
                    "Metode": method_name,
                    "Iterasi": folder,
                    "Kategori": cat_label,
                    "Jumlah Kalimat": int(data["total"]),
                    "Entropy Plain": data["entropy"][0],
                    "Entropy Cipher": data["entropy"][1],
                    "Avalanche (%)": data["avalanche"],
                    "Mean Time (µs)": mean_time,
                    "Total Time (µs)": total_time,
                }
            )
            times.setdefault(cat_label, []).append(exec_times)

            del data, items  # lepaskan memori sebelum file berikutnya

    return pd.DataFrame(summary_data), times


def mean_per_sentence(arrays):
    # Rata-rata waktu tiap kalimat lintas iterasi (indeks ke-i = kalimat yang sama)
    if len({a.size for a in arrays}) != 1:
        raise ValueError(
            "Jumlah kalimat antar iterasi berbeda; data tidak dapat dipasangkan."
        )
    return np.mean(np.vstack(arrays), axis=0)


def hitung_kenaikan(tabel, cat_order, judul, log_dict):
    # Hitung persentase kenaikan waktu antar kategori berurutan untuk tiap metode
    for method in METHODS:
        if method not in tabel.index or not all(c in tabel.columns for c in cat_order):
            continue

        t_1, t_2, t_3 = (tabel.loc[method, c] for c in cat_order)

        inc_1 = ((t_2 - t_1) / t_1) * 100 if pd.notna(t_1) and t_1 != 0 else 0
        inc_2 = ((t_3 - t_2) / t_2) * 100 if pd.notna(t_2) and t_2 != 0 else 0

        print(f"Metode {method}{judul}:")
        print(f"  - {cat_order[0]} -> {cat_order[1]} : Kenaikan {inc_1:.2f}%")
        print(f"  - {cat_order[1]} -> {cat_order[2]} : Kenaikan {inc_2:.2f}%")

        log_dict["Kenaikan Waktu"][method] = {
            f"{cat_order[0]} -> {cat_order[1]}": f"{inc_1:.2f}%",
            f"{cat_order[1]} -> {cat_order[2]}": f"{inc_2:.2f}%",
        }


def hitung_linearitas(tabel, n_per_kat, cat_order, log_dict):
    # Bandingkan kenaikan total waktu dengan kenaikan jumlah data.
    # Rasio (waktu/waktu awal) / (data/data awal) = 1 berarti tepat linear,
    # > 1 lebih buruk dari linear, < 1 lebih baik dari linear.
    print("\nLINEARITAS (rasio waktu / rasio data, 1.0 = linear)")
    log_dict["Linearitas"] = {}
    n0 = n_per_kat[cat_order[0]]

    for method in METHODS:
        if method not in tabel.index:
            continue
        t0 = tabel.loc[method, cat_order[0]]
        if pd.isna(t0) or t0 == 0:
            continue

        hasil = {}
        for c in cat_order[1:]:
            rasio_t = float(tabel.loc[method, c] / t0)
            rasio_n = n_per_kat[c] / n0
            hasil[f"{cat_order[0]} -> {c}"] = {
                "Rasio waktu": round(rasio_t, 3),
                "Rasio data": round(rasio_n, 3),
                "Rasio waktu / rasio data": round(rasio_t / rasio_n, 3),
            }
            print(
                f"Metode {method}, {cat_order[0]} -> {c}: waktu x{rasio_t:.2f}, "
                f"data x{rasio_n:.2f}, rasio = {rasio_t / rasio_n:.3f}"
            )
        log_dict["Linearitas"][method] = hasil


def buat_data_entropy(df_summary, log_dict):
    # Plaintext identik untuk kedua metode dan semua iterasi, jadi 1 bar per kategori.
    # Konsistensi dicek lintas SEMUA baris (metode x iterasi) per kategori.
    plain = (
        df_summary.groupby("Kategori", observed=True)["Entropy Plain"]
        .agg(nilai="mean", selisih=lambda s: s.max() - s.min())
        .reset_index()
    )

    if plain["selisih"].max() > 1e-6:
        pesan = "Entropy plaintext berbeda antar metode/iterasi, periksa data input."
        print(f"[Peringatan] {pesan}")
        log_dict["Peringatan Entropy"] = pesan

    df_plain = pd.DataFrame(
        {
            "Kategori": plain["Kategori"],
            "Nilai Entropy": plain["nilai"],
            "Grup": "Plaintext",
        }
    )

    df_cipher = pd.DataFrame(
        {
            "Kategori": df_summary["Kategori"],
            "Nilai Entropy": df_summary["Entropy Cipher"],
            "Grup": df_summary["Metode"] + " (Cipher)",
        }
    )

    df_entropy = pd.concat([df_plain, df_cipher], ignore_index=True)
    grup_order = [
        g
        for g in ["Plaintext", "Cross (Cipher)", "Sequenced (Cipher)"]
        if g in df_entropy["Grup"].unique()
    ]
    return df_entropy, grup_order


# SENSITIVITAS GRAFIK 1: boxplot sebaran waktu per kalimat
# (variabelnya adalah panjang tiap item, jadi sebaran per item memang yang diukur)
def grafik_boxplot_sensitivitas(per_sentence, cat_order, scenario_name):
    rng = np.random.default_rng(SEED)
    rows = []
    for method, d in per_sentence.items():
        for cat, arr in d.items():
            sample = (
                arr
                if arr.size <= MAX_BOX_POINTS
                else rng.choice(arr, MAX_BOX_POINTS, replace=False)
            )
            rows.append(
                pd.DataFrame({"Metode": method, "Kategori": cat, "Waktu (µs)": sample})
            )
    df_time = pd.concat(rows, ignore_index=True)
    df_time["Kategori"] = pd.Categorical(
        df_time["Kategori"], categories=cat_order, ordered=True
    )

    plt.figure(figsize=(8.27, 6))
    sns.boxplot(
        data=df_time,
        x="Kategori",
        y="Waktu (µs)",
        hue="Metode",
        hue_order=METHODS,
        palette=PALETTE,
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 6,
        },
    )
    plt.title(f"Distribusi Waktu Eksekusi per Kalimat ({scenario_name})")
    plt.tight_layout()
    simpan_grafik(f"grafik_1_waktu_{scenario_name.lower()}.png")


# SENSITIVITAS GRAFIK 2: trend rata-rata waktu per kalimat antar kategori panjang
def grafik_trend_sensitivitas(df_summary, scenario_name):
    plt.figure(figsize=(8.27, 6))
    sns.lineplot(
        data=df_summary,
        x="Kategori",
        y="Mean Time (µs)",
        hue="Metode",
        hue_order=METHODS,
        palette=PALETTE,
        marker="o",
        linewidth=2.5,
        errorbar="sd",
        err_style="bars",
    )
    plt.title(f"Trend Waktu Eksekusi per Kalimat ({scenario_name})")
    plt.ylabel("Rata-rata Waktu per Kalimat (µs)")
    plt.tight_layout()
    simpan_grafik(f"grafik_2_trend_{scenario_name.lower()}.png")


# SKALABILITAS GRAFIK 1: total waktu terhadap jumlah kalimat SEBENARNYA,
# dengan garis putus-putus = skala linear ideal dari titik terkecil
def grafik_total_skalabilitas(df_summary, scenario_name):
    plt.figure(figsize=(8.27, 6))
    ax = sns.lineplot(
        data=df_summary,
        x="Jumlah Kalimat",
        y="Total Time (s)",
        hue="Metode",
        hue_order=METHODS,
        palette=PALETTE,
        marker="o",
        linewidth=2.5,
        errorbar="sd",
        err_style="bars",
    )

    n_values = sorted(df_summary["Jumlah Kalimat"].unique())
    n_ref = n_values[0]
    for method in METHODS:
        sub = df_summary[
            (df_summary["Metode"] == method) & (df_summary["Jumlah Kalimat"] == n_ref)
        ]
        if sub.empty:
            continue
        t_ref = sub["Total Time (s)"].mean()
        ax.plot(
            n_values,
            [t_ref * n / n_ref for n in n_values],
            linestyle="--",
            linewidth=1.5,
            color=PALETTE[method],
            alpha=0.6,
            label=f"{method} (linear ideal)",
        )

    ax.set_xticks(n_values)
    ax.set_xticklabels([format_n(n) for n in n_values])
    plt.xlabel("Jumlah Kalimat")
    plt.ylabel("Total Waktu Eksekusi (detik)")
    plt.title(f"Total Waktu terhadap Jumlah Data ({scenario_name})")
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    simpan_grafik(f"grafik_1_total_{scenario_name.lower()}.png")


# SKALABILITAS GRAFIK 2: rata-rata waktu per kalimat terhadap jumlah kalimat
# (garis mendatar = biaya per kalimat konstan; sumbu y dimulai dari 0)
def grafik_per_kalimat_skalabilitas(df_summary, scenario_name):
    plt.figure(figsize=(8.27, 6))
    ax = sns.lineplot(
        data=df_summary,
        x="Jumlah Kalimat",
        y="Mean Time (µs)",
        hue="Metode",
        hue_order=METHODS,
        palette=PALETTE,
        marker="o",
        linewidth=2.5,
        errorbar="sd",
        err_style="bars",
    )
    n_values = sorted(df_summary["Jumlah Kalimat"].unique())
    ax.set_xticks(n_values)
    ax.set_xticklabels([format_n(n) for n in n_values])
    ax.set_ylim(bottom=0)
    plt.xlabel("Jumlah Kalimat")
    plt.ylabel("Rata-rata Waktu per Kalimat (µs)")
    plt.title(f"Trend Rata-rata Waktu per Kalimat ({scenario_name})")
    plt.tight_layout()
    simpan_grafik(f"grafik_2_perkalimat_{scenario_name.lower()}.png")


def run_analysis(scenario_name, cross_dir, seq_dir, is_sensitivity):
    log_dict = {"Skenario": scenario_name, "T-Test": {}, "Kenaikan Waktu": {}}

    print(f"\nMEMULAI ANALISIS: {scenario_name.upper()}")

    df_summary_cross, times_cross = load_data(cross_dir, "Cross", is_sensitivity)
    df_summary_seq, times_seq = load_data(seq_dir, "Sequenced", is_sensitivity)

    if df_summary_cross.empty and df_summary_seq.empty:
        print(
            f"Melewati {scenario_name}: Data tidak ditemukan di direktori yang dituju."
        )
        log_dict["Status"] = "Data tidak ditemukan"
        save_log("hasil", log_dict)
        return

    df_summary = pd.concat([df_summary_cross, df_summary_seq], ignore_index=True)

    if is_sensitivity:
        cat_order = ["Pendek", "Menengah", "Panjang"]
    else:
        cat_order = ["Sedikit", "Sedang", "Banyak"]

    df_summary["Kategori"] = pd.Categorical(
        df_summary["Kategori"], categories=cat_order, ordered=True
    )

    # Rata-rata per kalimat lintas iterasi (menghindari pseudo-replikasi)
    per_sentence = {
        "Cross": {c: mean_per_sentence(a) for c, a in times_cross.items()},
        "Sequenced": {c: mean_per_sentence(a) for c, a in times_seq.items()},
    }
    log_dict["Jumlah Iterasi"] = {
        m: sorted({len(a) for a in t.values()})
        for m, t in (("Cross", times_cross), ("Sequenced", times_seq))
    }

    # UJI-T INDEPENDEN (Welch) pada rata-rata waktu per kalimat antar iterasi.
    # n = jumlah kalimat (bukan kalimat x iterasi), sehingga tidak ada pseudo-replikasi.
    # Uji berpasangan dicatat di log hanya sebagai pembanding.
    print(
        "\nHASIL INDEPENDENT T-TEST / WELCH (WAKTU EKSEKUSI PER KALIMAT, RATA-RATA ANTAR ITERASI)"
    )

    for cat in cat_order:
        a = per_sentence["Cross"].get(cat)
        b = per_sentence["Sequenced"].get(cat)
        if a is None or b is None:
            continue

        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        signifikansi = (
            "Signifikan (H1 diterima)"
            if p_val < 0.05
            else "Tidak Signifikan (H0 diterima)"
        )
        selisih = float(a.mean() - b.mean())
        sd_gabungan = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
        d = selisih / sd_gabungan if sd_gabungan > 0 else 0.0  # Cohen's d

        print(f"Kategori {cat}:")
        print(f"  - n (Cross)            : {a.size}")
        print(f"  - n (Sequenced)        : {b.size}")
        print(f"  - Rata-rata Cross      : {a.mean():.4f} µs")
        print(f"  - Rata-rata Sequenced  : {b.mean():.4f} µs")
        print(f"  - Selisih (Cross-Seq)  : {selisih:.4f} µs")
        print(f"  - t-value              : {t_stat:.4f}")
        print(f"  - p-value              : {p_val:.4e} -> {signifikansi}")
        print(f"  - Effect size (Cohen d): {d:.4f}")

        entri = {
            "n (Cross)": int(a.size),
            "n (Sequenced)": int(b.size),
            "Rata-rata Cross (µs)": round(float(a.mean()), 4),
            "Rata-rata Sequenced (µs)": round(float(b.mean()), 4),
            "Selisih Cross-Seq (µs)": round(selisih, 4),
            "t-value": round(float(t_stat), 4),
            "p-value": f"{p_val:.4e}",
            "Cohen d": round(d, 4),
            "Hasil": signifikansi,
        }
        if a.size == b.size:
            t_rel, p_rel = stats.ttest_rel(a, b)
            entri["Pembanding berpasangan"] = {
                "t-value": round(float(t_rel), 4),
                "p-value": f"{p_rel:.4e}",
            }
        log_dict["T-Test"][cat] = entri

    # ANALISIS KENAIKAN WAKTU + GRAFIK WAKTU (berbeda per skenario)
    print("\nANALISIS KENAIKAN WAKTU")

    if is_sensitivity:
        # Variabel = panjang item -> bandingkan rata-rata waktu per kalimat antar kategori
        tabel = (
            df_summary.groupby(["Metode", "Kategori"], observed=False)["Mean Time (µs)"]
            .mean()
            .unstack()
        )
        hitung_kenaikan(tabel, cat_order, "", log_dict)

        grafik_boxplot_sensitivitas(per_sentence, cat_order, scenario_name)
        grafik_trend_sensitivitas(df_summary, scenario_name)
    else:
        # Variabel = jumlah item -> bandingkan total waktu (detik), dirata-rata antar iterasi
        df_summary["Total Time (s)"] = df_summary["Total Time (µs)"] / 1e6
        tabel = (
            df_summary.groupby(["Metode", "Kategori"], observed=False)["Total Time (s)"]
            .mean()
            .unstack()
        )
        hitung_kenaikan(tabel, cat_order, " (Total)", log_dict)

        n_per_kat = df_summary.groupby("Kategori", observed=True)[
            "Jumlah Kalimat"
        ].first()
        hitung_linearitas(tabel, n_per_kat, cat_order, log_dict)

        grafik_total_skalabilitas(df_summary, scenario_name)
        grafik_per_kalimat_skalabilitas(df_summary, scenario_name)

    # GRAFIK 3: ENTROPY (deterministik -> tanpa error bar)
    plt.figure(figsize=(8.27, 6))
    df_entropy, grup_order = buat_data_entropy(df_summary, log_dict)
    sns.barplot(
        data=df_entropy,
        x="Kategori",
        y="Nilai Entropy",
        hue="Grup",
        hue_order=grup_order,
        errorbar=None,
    )
    plt.ylim(0, 7)  # ruang atas agar legenda tidak menutupi batang
    plt.title(f"Entropy Plaintext vs Ciphertext ({scenario_name})")
    plt.legend(title="Data", fontsize=8, loc="upper right")
    plt.tight_layout()
    simpan_grafik(f"grafik_3_entropy_{scenario_name.lower()}.png")

    # GRAFIK 4: AVALANCHE (deterministik -> tanpa error bar; garis 50% = acuan ideal)
    plt.figure(figsize=(8.27, 6))
    sns.barplot(
        data=df_summary,
        x="Kategori",
        y="Avalanche (%)",
        hue="Metode",
        hue_order=METHODS,
        palette=PALETTE,
        errorbar=None,
    )
    plt.axhline(50, color="red", linestyle="--", linewidth=1, label="Ideal (50%)")
    plt.title(f"Perbandingan Avalanche Effect ({scenario_name})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    simpan_grafik(f"grafik_4_avalanche_{scenario_name.lower()}.png")

    print(f"[Sukses] Grafik untuk {scenario_name} disimpan di: {FIGURES_DIR}")

    save_log("hasil", log_dict)


def main():
    # Skalabilitas (Scale)
    run_analysis(
        scenario_name="Skalabilitas",
        cross_dir=input_dir_scale_group1,
        seq_dir=input_dir_scale_group2,
        is_sensitivity=False,
    )

    # Sensitivitas (Sense)
    run_analysis(
        scenario_name="Sensitivitas",
        cross_dir=input_dir_sense_group1,
        seq_dir=input_dir_sense_group2,
        is_sensitivity=True,
    )


if __name__ == "__main__":
    main()
