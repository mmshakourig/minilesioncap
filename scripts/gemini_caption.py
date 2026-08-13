"""
Turn a lesion_metrics.py row into a standard-format text prompt and ask a free-tier
Gemini model (Google AI Studio, generateContent REST endpoint) for a free-text
radiology-style description of the lesion.

API key resolution order: $GEMINI_API_KEY env var, then a GEMINI_API_KEY=... line
in a .env file at the repo root.

Usage:
    python scripts/gemini_caption.py --patient-dir data/11-37493
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gemini-flash-lite-latest"  # free-tier alias, no extra "thinking" token overhead
GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def load_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    raise RuntimeError(
        "No Gemini API key found. Set GEMINI_API_KEY as an environment variable, "
        "or put GEMINI_API_KEY=<your key> in a .env file at the repo root."
    )


def describe_relative_position(lesion_xyz, organ_xyz, threshold_mm: float = 5.0) -> str:
    """
    DICOM patient space is LPS: +x = patient Left, +y = Posterior, +z = Superior.
    Turn the lesion-minus-organ centroid offset into a phrase like
    "2.1 cm anterior, 0.8 cm superior, 1.4 cm to the right".
    """
    if organ_xyz is None or any(v is None for v in organ_xyz):
        return "unknown (closest organ centroid unavailable)"

    dx, dy, dz = (lesion_xyz[i] - organ_xyz[i] for i in range(3))
    parts = []
    if abs(dx) >= threshold_mm:
        parts.append(f"{abs(dx) / 10:.1f} cm {'left' if dx > 0 else 'right'}")
    if abs(dy) >= threshold_mm:
        parts.append(f"{abs(dy) / 10:.1f} cm {'posterior' if dy > 0 else 'anterior'}")
    if abs(dz) >= threshold_mm:
        parts.append(f"{abs(dz) / 10:.1f} cm {'superior' if dz > 0 else 'inferior'}")
    return ", ".join(parts) if parts else "essentially at the same location"


def build_prompt(row: pd.Series) -> str:
    lesion_xyz = tuple(float(v) for v in (row["lesion_centroid_x_mm"], row["lesion_centroid_y_mm"], row["lesion_centroid_z_mm"]))
    organ_xyz_raw = (
        row.get("closest_anatomy_centroid_x_mm"),
        row.get("closest_anatomy_centroid_y_mm"),
        row.get("closest_anatomy_centroid_z_mm"),
    )
    organ_xyz = tuple(None if v is None or pd.isna(v) else float(v) for v in organ_xyz_raw)
    relative_position = describe_relative_position(lesion_xyz, organ_xyz)

    return f"""You are a nuclear medicine physician describing a lesion found on a whole-body PET/CT scan.
Coordinates are in the DICOM patient (LPS) frame, in millimeters: +x = Left, +y = Posterior, +z = Superior.

Lesion data:
- Lesion ID: {row['lesion']}
- Closest segmented organ/structure (TotalSegmentator): {row['closest_anatomy']}
- Distance from lesion to that structure's boundary: {row['distance_mm']:.1f} mm
- Lesion centroid (x, y, z) mm: ({lesion_xyz[0]:.1f}, {lesion_xyz[1]:.1f}, {lesion_xyz[2]:.1f})
- Closest organ centroid (x, y, z) mm: {tuple(round(v, 1) if v is not None else None for v in organ_xyz)}
- Lesion position relative to that organ's centroid: {relative_position}
- Axial (CT) slice index: {int(row['axial_slice_index'])}
- SUVmax: {row['suv_max']:.2f}
- SUVmean (within the whole lesion contour): {row['suv_mean_whole']:.2f}
- Metabolic Tumor Volume (MTV, 41% SUVmax threshold): {row['mtv_ml']:.2f} mL
- Total Lesion Glycolysis (TLG): {row['tlg_g']:.2f} g

Write a short (3-5 sentence) free-text radiology-style description of this lesion that:
1. States its location relative to the closest organ (using the distance and relative-position data above).
2. States its general anatomical region in standard radiology terms (e.g. "right upper abdomen",
   "left lower thorax", "lumbar spine"), inferred from the organ name and coordinates.
3. Reports the SUVmax.
4. Reports the Metabolic Tumor Volume (MTV) and Total Lesion Glycolysis (TLG).
Be concise and clinical. Do not invent findings not supported by the data above.
Respond in plain prose only: no markdown, no LaTeX, no bullet points, no headers."""


def call_gemini(prompt: str, api_key: str, model: str = DEFAULT_MODEL, timeout: int = 30, max_retries: int = 3) -> str:
    url = GENERATE_URL.format(model=model)
    body = {"contents": [{"parts": [{"text": prompt}]}]}

    for attempt in range(max_retries):
        response = requests.post(url, params={"key": api_key}, json=body, timeout=timeout)
        if response.status_code == 429 and attempt < max_retries - 1:
            time.sleep(2 ** attempt * 5)
            continue
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates")
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {data}")
        parts = candidates[0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()

    raise RuntimeError("Gemini API call failed after retries (rate limited)")


def generate_caption_for_lesion(row: pd.Series, api_key: str, model: str = DEFAULT_MODEL) -> str:
    prompt = build_prompt(row)
    return call_gemini(prompt, api_key, model)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patient-dir", type=Path, default=Path("data/11-37493"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to outputs/<patient-dir name>")
    parser.add_argument("--report-csv", type=Path, default=None, help="Defaults to <output-dir>/lesion_report.csv")
    parser.add_argument("--captions-out", type=Path, default=None, help="Defaults to <output-dir>/lesion_captions.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true", help="Regenerate captions even if already cached")
    args = parser.parse_args()

    output_dir = args.output_dir or Path("outputs") / args.patient_dir.name
    report_csv = args.report_csv or output_dir / "lesion_report.csv"
    captions_out = args.captions_out or output_dir / "lesion_captions.json"

    if not report_csv.exists():
        raise FileNotFoundError(f"{report_csv} not found. Run scripts/lesion_metrics.py first.")

    report = pd.read_csv(report_csv)
    api_key = load_api_key()

    captions = {}
    if captions_out.exists() and not args.force:
        captions = json.loads(captions_out.read_text())

    for _, row in report.iterrows():
        name = row["lesion"]
        if name in captions and not args.force:
            print(f"  {name}: already cached, skipping")
            continue
        if pd.isna(row.get("suv_max")):
            print(f"  {name}: no PET metrics available (outside CT FOV), skipping")
            continue
        print(f"  {name}: requesting caption from {args.model}...")
        try:
            captions[name] = generate_caption_for_lesion(row, api_key, args.model)
        except Exception as exc:
            print(f"    failed: {exc}")
            continue

        captions_out.parent.mkdir(parents=True, exist_ok=True)
        captions_out.write_text(json.dumps(captions, indent=2))

    print(f"\nWrote {len(captions)} captions to {captions_out}")


if __name__ == "__main__":
    main()
