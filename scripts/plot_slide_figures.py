"""Numerical figure exports and checks against archived chart values.

Frozen chart values provide the verification baseline. When a presentation is
supplied separately, its illustrations and screenshots can also be exported."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data/reference"
RESULTS = ROOT / "results/bloch_corrected"
OUTPUT = ROOT / "figures"


def step(number):
    return next(RESULTS.glob(f"step_{number}_*"))


def table(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    fig.savefig(OUTPUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def inverse_sqrt_fit(x, y):
    # Retain both grid-search step sizes to match the final chart fit.
    def candidate(b):
        p = 1 / np.sqrt(x + b)
        a, c = np.linalg.lstsq(np.column_stack([p, np.ones(len(x))]), y, rcond=None)[0]
        return float(np.mean((a * p + c - y) ** 2)), float(b), float(a), float(c)

    best = min(candidate(b) for b in np.arange(0, 20.0001, 0.01))
    best = min(candidate(b) for b in np.arange(max(0, best[1] - 0.03), best[1] + 0.03001, 0.0001))
    return best[1:]


def downsample(values, stride):
    idx = list(range(0, len(values), stride))
    if idx[-1] != len(values) - 1:
        idx.append(len(values) - 1)
    return values[idx]


def export_embedded_assets(deck_path):
    ns = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    }
    asset_dir = OUTPUT / "embedded"
    asset_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(deck_path) as deck:
        for name in deck.namelist():
            if name.startswith("ppt/media/"):
                (asset_dir / Path(name).name).write_bytes(deck.read(name))
        manifest = []
        relns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        rels = {
            r.get("Id"): r.get("Target")
            for r in ET.fromstring(deck.read("ppt/_rels/presentation.xml.rels"))
        }
        presentation = ET.fromstring(deck.read("ppt/presentation.xml"))
        for i, sid in enumerate(presentation.findall(".//p:sldId", ns), 1):
            slidefile = "ppt/" + rels[sid.get("{" + relns + "}id")]
            xml = ET.fromstring(deck.read(slidefile))
            part = Path(slidefile).name
            slide_rels = ET.fromstring(deck.read(f"ppt/slides/_rels/{part}.rels"))
            manifest.append(
                {
                    "slide": i,
                    "xml_part": part,
                    "text": " ".join(t.text or "" for t in xml.findall(".//a:t", ns)),
                    "assets": sorted(
                        {
                            Path(r.get("Target")).name
                            for r in slide_rels
                            if r.get("Type", "").endswith("/image")
                        }
                    ),
                    "note": "Includes off-canvas/hidden assets; native shapes remain in the PPTX.",
                }
            )
    (OUTPUT / "embedded_assets.json").write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--deck",
        type=Path,
        help="optional presentation used only to export embedded media and record its checksum",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    chart = json.loads((REF / "final_slide_chart_series.json").read_text())
    checks = {}

    extra = ROOT / "results/extended_stubs.csv"
    pts = rows(step(39) / "01_fitted_lengths.csv") + rows(
        extra if extra.exists() else REF / "hfss_40_50_60um_fits.csv"
    )
    pts = sorted(pts, key=lambda r: float(r["stub_length_um"]))
    x = np.array([float(r["stub_length_um"]) for r in pts])
    y = np.array([float(r["Z0_ohm"]) for r in pts])
    b, a, c = inverse_sqrt_fit(x, y)
    xf = np.arange(2.5, 60.0001, 0.25)
    yf = a / np.sqrt(xf + b) + c
    paper = table(step(32) / "01b_paper_digitized_Z0_curve.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(paper[:, 0], paper[:, 1], color="#5B95DE", label="Paper")
    ax.scatter(x, y, color="#ED6A2C", label="HFSS points", zorder=3)
    ax.plot(xf, yf, color="#ED6A2C", label="HFSS fit")
    ax.set(
        xlim=(2.5, 60),
        ylim=(15, 125),
        xlabel="Stub length (µm)",
        ylabel="Characteristic impedance, Z₀ (Ω)",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, "slide_13_stub_impedance")
    checks["impedance_point_max_error_ohm"] = float(np.max(np.abs(y - chart["1"][1]["y"])))
    checks["impedance_fit_max_error_ohm"] = float(np.max(np.abs(yf - chart["1"][2]["y"])))

    # Accumulated Bloch evanescence is the slide's transmission proxy.
    disp = table(step(34) / "01_dispersion.csv")
    bare = table(step(34) / "01b_dispersion_bare_reference.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    axes[0].plot(disp[:, 0], disp[:, 1], color="black")
    axes[0].set(
        xlim=(1, 27),
        ylim=(-80, 5),
        xlabel="Frequency [GHz]",
        ylabel="|S21| [dB]",
        title="(a) HFSS-simulated frequency range",
    )
    axes[1].plot(bare[:, 0], bare[:, 1] * 1e3, "--", color="gray", label="bare (theoretical)")
    axes[1].plot(disp[:, 0], disp[:, 2] * 1e3, color="black", label="disp. eng. (HFSS-derived)")
    for edge in (1, 27):
        axes[1].axvline(edge, color="#2a78d6", ls=":", lw=0.8)
    axes[1].set(xlim=(0, 30), xlabel="Frequency [GHz]", ylabel="k* [mrad]", title="(b)")
    axes[1].legend(fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.25)
    save(fig, "slide_16_passive_response")

    # Mask idlers outside the simulated band instead of extrapolating gain.
    heatfile = step(34) / "02c_gain_heatmap_extended_x14.csv"
    heat = table(heatfile)
    phase = table(step(34) / "02d_phase_mismatch_extended_x14.csv")
    pumps = np.array(
        [
            float(re.search(r"pump_([\d.]+)GHz", h)[1])
            for h in heatfile.read_text().splitlines()[0].split(",")[1:]
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("black")
    mesh = ax.pcolormesh(
        heat[:, 0],
        pumps,
        np.ma.masked_invalid(heat[:, 1:].T),
        cmap=cmap,
        vmin=0,
        vmax=40,
        shading="auto",
    )
    ax.contour(
        phase[:, 0],
        pumps,
        np.ma.masked_invalid(phase[:, 1:].T),
        levels=[0],
        colors=["#55e0ff"],
        linewidths=1.5,
    )
    ax.plot([], [], color="#55e0ff", label=r"$\Delta k_\mathrm{eff}=0$")
    ax.legend(facecolor="black", labelcolor="white", loc="upper right")
    fig.colorbar(mesh, ax=ax, location="top", label="Gain [dB]")
    ax.set(
        xlim=(2.5, 14),
        ylim=(12.5, 17.5),
        xlabel="Signal frequency [GHz]",
        ylabel="Pump frequency [GHz]",
    )
    save(fig, "slide_17_gain_map")

    # The final comparison uses the bandwidth-matched pump, not the GBP optimum.
    paper_dir = ROOT / "results/paper_curves"
    experiment = table((paper_dir if paper_dir.exists() else REF) / "paper_experiment_fig4a.csv")
    simulation = table((paper_dir if paper_dir.exists() else REF) / "paper_simulation_fig3d.csv")
    model = table(ROOT / "results/refined_pump/refined_model_gain_profile.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    for data, label, color in [
        (downsample(experiment, 4), "Paper experiment", "#f000f0"),
        (simulation, "Paper simulation", "gray"),
        (downsample(model, 5), "Our model (13.865 GHz)", "#1775b5"),
    ]:
        ax.plot(data[:, 0], data[:, 1], color=color, label=label, lw=1.8)
    ax.set(xlim=(2, 11), ylim=(-10, 45), xlabel="Signal frequency [GHz]", ylabel="Gain [dB]")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save(fig, "slide_18_gain_comparison")
    for label, data, ref in [
        ("experiment", downsample(experiment, 4), chart["2"][0]),
        ("simulation", simulation, chart["2"][1]),
        ("model", downsample(model, 5), chart["2"][2]),
    ]:
        checks[label + "_chart_max_error"] = float(
            np.max(np.abs(data - np.column_stack([ref["x"], ref["y"]])))
        )
    checks["model_peak_gain_dB"] = float(np.max(model[:, 1]))
    refined = json.loads((ROOT / "results/refined_pump/refined_result.json").read_text())["best"]
    checks["selected_pump_GHz"] = refined["pump_GHz"]
    checks["bandwidth_3dB_GHz"] = refined["B3dB_bandwidth_GHz"]

    # Each device is evaluated in its own mid-band frequency window.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, number, name, title, color in [
        (axes[0], 37, "06_fine_numeric_ripple_check.csv", "Baseline device", "tab:blue"),
        (axes[1], 41, "03_fine_numeric_ripple_check.csv", "Stub-only redesign", "tab:green"),
    ]:
        data = table(step(number) / name)
        ripple = float(np.ptp(data[:, 1]))
        mean = float(np.mean(data[:, 1]))
        ax.plot(data[:, 0], data[:, 1], color=color, lw=1.2)
        ax.axhline(mean, color="gray", ls=":", label=f"Mean = {mean:.2f} dB")
        ax.set(
            title=f"{title}: {ripple:.2f} dB peak-to-peak",
            xlabel="Signal frequency [GHz]",
            ylabel="Gain [dB]",
        )
        ax.grid(alpha=0.25)
        ax.legend()
        checks[f"step_{number}_ripple_dB"] = ripple
    save(fig, "slide_25_reflection_ripple")
    if args.deck:
        export_embedded_assets(args.deck)
        checks["deck_sha256"] = hashlib.sha256(args.deck.read_bytes()).hexdigest()
    checks["missing_raw_exports"] = [
        f"kinetic_{n}um_{band}.s2p"
        for n in (40, 50, 60)
        for band in ("debug", "lowfreq")
        if not (ROOT / "hfss_inputs" / f"kinetic_{n}um_{band}.s2p").exists()
    ]
    (OUTPUT / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    if args.verify:
        assert checks["impedance_point_max_error_ohm"] < 1e-6, checks
        assert checks["impedance_fit_max_error_ohm"] < 1e-5, checks
        for label in ("experiment", "simulation", "model"):
            assert checks[label + "_chart_max_error"] < 1e-5, checks
        assert abs(checks["model_peak_gain_dB"] - 33.74288885209461) < 1e-4, checks
        assert abs(checks["selected_pump_GHz"] - 13.865) < 1e-8, checks
        assert abs(checks["bandwidth_3dB_GHz"] - 3.16) < 1e-8, checks
        assert abs(checks["step_37_ripple_dB"] - 4.10) < 0.01, checks
        assert abs(checks["step_41_ripple_dB"] - 2.74) < 0.01, checks
        print("Final-slide numerical checks passed.")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
