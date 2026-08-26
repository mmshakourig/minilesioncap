"""
SUV Conversion Pipeline — IBSI-SUV compliant
=============================================
Implements every rule from the IBSI-SUV manual (Vácha et al. 2026) and
applies them to the DROs from oncoray/suv_computation.

Fixes vs. original QURIT NIFTI_conversion.py
---------------------------------------------
BUG-1  Rescale slope was applied once per series; must be applied per-slice.
BUG-2  Units = GML not handled (SUVbw / SUVlbm / SUVibw already stored).
BUG-3  Units = CM2ML (SUVbsa) not handled.
BUG-4  Units = CNTS (Philips SUV & activity scale factors) not handled.
BUG-5  Dose unit detection missing: values < 1e4 Bq → assumed MBq, convert.
BUG-6  DecayCorrection = ADMIN not handled (no extra decay correction needed).
BUG-7  DecayCorrection = NONE not handled.
BUG-8  Reference-time for DC=START: SeriesTime vs AcquisitionTime not compared;
        Option-4 back-computation from AcqTime + FrameReferenceTime + FrameDuration
        not implemented.
BUG-9  GE private tag (0009,100D) for scan DateTime not checked.
BUG-10 RadiopharmaceuticalStartDateTime (0018,1078) not tried when
        RadiopharmaceuticalStartTime (0018,1072) is absent → crash on DRO_4_0.
BUG-11 Midnight-crossing not handled for administration time.
BUG-12 LBM (LBMJAMES128 / LBM / LBMJANMA) and IBW back-conversion missing.
BUG-13 PatientSex = "O" not handled (use mean of male/female factors).

Updates for IBSI-SUV DRO v2 (expanded 17 → 32 DROs)
-----------------------------------------------------
FIX-14 Siemens private decay-correction datetime tag (0071,1022) not handled;
        added as Priority 2 in _get_reference_time_start(), after GE tag but
        before Option 4, covering DRO_3_3_0.
        Priority order is now: GE tag → Siemens tag → Option 4 → SeriesTime.
FIX-16 FrameReferenceTime = 0 causes Option 4 to overcorrect by +Tave (~75s for
        F18, ~38s for Ga68). When FRT=0 the scanner records tref = AcquisitionTime
        exactly (the frame starts at scan time, no midpoint offset). Fix: skip
        Option 4 when FRT=0 and use AcquisitionTime directly as tref instead.
        Covers DRO_3_5_0/1/2 (Siemens/GE/Philips, SeriesTime=AcquisitionTime).
        The CNTS block previously always applied DC=START after ac_sf conversion.
        It now falls through to the correct DC path (ADMIN / NONE / START),
        enabling the per-frame decay correction to run when DC=NONE.

Requirements
------------
    pip install pydicom SimpleITK rt-utils python-dateutil numpy nibabel
"""

import os
import subprocess
import sys
import math
from datetime import datetime, timedelta

import numpy as np
import pydicom
import SimpleITK as sitk
import nibabel as nib

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

REPO_URL = "https://github.com/oncoray/suv_computation.git"
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "suv_computation")
DRO_ROOT  = os.path.join(REPO_DIR, "DRO")


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY: date/time parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_dicom_datetime(date_str: str, time_str: str) -> datetime:
    """Parse DICOM date (YYYYMMDD) + time (HHMMSS[.ffffff]) into a datetime."""
    date_str = str(date_str).strip()
    time_str = str(time_str).strip().split(".")[0]
    return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")


def _parse_dicom_dt_string(dt_str: str) -> datetime:
    """Parse a DICOM DT string 'YYYYMMDDHHMMSS[.ffffff][+-tz]' into a datetime."""
    dt_str = str(dt_str).strip()
    for sep in ("+", "-"):
        if sep in dt_str[8:]:
            dt_str = dt_str[: dt_str.index(sep, 8)]
    dt_str = dt_str.split(".")[0]
    return datetime.strptime(dt_str, "%Y%m%d%H%M%S")


def _get_administration_datetime(radiopharm_item, series_date_str: str) -> datetime:
    """
    FIX BUG-10: try RadiopharmaceuticalStartDateTime (0018,1078) first,
    then fall back to RadiopharmaceuticalStartTime (0018,1072).
    """
    if (0x0018, 0x1078) in radiopharm_item:
        raw = str(radiopharm_item[0x0018, 0x1078].value).strip()
        if raw:
            return _parse_dicom_dt_string(raw)

    if (0x0018, 0x1072) not in radiopharm_item:
        raise KeyError(
            "Neither RadiopharmaceuticalStartDateTime (0018,1078) nor "
            "RadiopharmaceuticalStartTime (0018,1072) found."
        )

    inj_time_str = str(radiopharm_item[0x0018, 0x1072].value).strip().split(".")[0]
    return _parse_dicom_datetime(series_date_str, inj_time_str)


def _midnight_corrected(admin_dt: datetime, ref_dt: datetime) -> datetime:
    """FIX BUG-11: if admin appears after ref, injection crossed midnight."""
    if admin_dt > ref_dt:
        admin_dt -= timedelta(days=1)
    return admin_dt


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY: reference-time computation for DC = START
# ──────────────────────────────────────────────────────────────────────────────

def _compute_reference_time_option4(
    acq_date_str: str,
    acq_time_str: str,
    frame_ref_time_ms: float,
    frame_duration_ms: float,
    half_life_s: float,
) -> datetime:
    """
    FIX BUG-8 (IBSI manual Option B / QIBA Option 4):
    tref = tacq + Tave - delta_t
    Tave = (1/lam) * ln( lam*T / (1 - exp(-lam*T)) )
    """
    lam     = math.log(2) / half_life_s
    T       = frame_duration_ms / 1000.0
    delta_t = frame_ref_time_ms / 1000.0
    Tave    = (1.0 / lam) * math.log(lam * T / (1.0 - math.exp(-lam * T))) if T > 0 else 0.0
    tacq    = _parse_dicom_datetime(acq_date_str, acq_time_str)
    return tacq + timedelta(seconds=(Tave - delta_t))


def _get_reference_time_start(first_dcm, half_life_s: float) -> datetime:
    """
    FIX BUG-8/9/14: determine tref for DC=START.

    Priority order (IBSI manual recommendation):
      1. GE private tag (0009,100D)   — GE scanners use this as their actual
                                        decay-correction anchor; must take priority
                                        over Option 4 when present.
      2. Siemens private tag (0071,1022) — Siemens decay correction datetime;
                                        always equals SeriesTime for Siemens but
                                        explicit use makes the intent clear.
      3. Option 4 back-computation    — from AcquisitionTime + FrameReferenceTime
                                        + FrameDuration; vendor-agnostic and robust
                                        to SeriesTime post-processing modifications.
      4. SeriesTime                   — final fallback.
    """
    series_date = str(first_dcm[0x0008, 0x0021].value).strip()
    series_time = str(first_dcm[0x0008, 0x0031].value).strip().split(".")[0]

    # Priority 1: GE private scan datetime (FIX BUG-9)
    if (0x0009, 0x100D) in first_dcm:
        raw = str(first_dcm[0x0009, 0x100D].value).strip()
        if raw:
            return _parse_dicom_dt_string(raw)

    # Priority 2: Siemens private decay correction datetime (FIX-14)
    if (0x0071, 0x1022) in first_dcm:
        raw = str(first_dcm[0x0071, 0x1022].value).strip()
        if raw:
            return _parse_dicom_dt_string(raw)

    # Priority 3: Option 4 back-computation (FIX BUG-8)
    # Guard: FrameReferenceTime must be > 0.
    # When FRT = 0 the frame starts at AcquisitionTime, meaning tref = AcquisitionTime
    # exactly. Applying Option 4 with FRT=0 incorrectly adds Tave (~75 s for F18)
    # on top of AcquisitionTime and overcorrects the dose → wrong SUV.
    # In that case we fall through to the SeriesTime fallback (which equals
    # AcquisitionTime for these DROs). (FIX-16)
    has_frt = (0x0054, 0x1300) in first_dcm
    has_fd  = (0x0018, 0x1242) in first_dcm
    has_acq = (0x0008, 0x0032) in first_dcm and (0x0008, 0x0022) in first_dcm
    frt_nonzero = has_frt and float(first_dcm[0x0054, 0x1300].value) > 0.0

    if has_frt and has_fd and has_acq and frt_nonzero:
        acq_date     = str(first_dcm[0x0008, 0x0022].value).strip()
        acq_time     = str(first_dcm[0x0008, 0x0032].value).strip().split(".")[0]
        frame_ref_ms = float(first_dcm[0x0054, 0x1300].value)
        frame_dur_ms = float(first_dcm[0x0018, 0x1242].value)
        return _compute_reference_time_option4(
            acq_date, acq_time, frame_ref_ms, frame_dur_ms, half_life_s
        )

    # Priority 4: AcquisitionTime when FRT=0 (frame start = tref)  (FIX-16)
    if has_acq and has_frt and not frt_nonzero:
        acq_date = str(first_dcm[0x0008, 0x0022].value).strip()
        acq_time = str(first_dcm[0x0008, 0x0032].value).strip().split(".")[0]
        return _parse_dicom_datetime(acq_date, acq_time)

    # Priority 5: SeriesTime fallback
    return _parse_dicom_datetime(series_date, series_time)


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY: body-composition normalization factors  (FIX BUG-12/13)
# ──────────────────────────────────────────────────────────────────────────────

def _lbm_james128(w_kg, h_cm, sex):
    if sex == "M":
        return 1.10 * w_kg - 128.0 * (w_kg / h_cm) ** 2
    if sex == "F":
        return 1.07 * w_kg - 148.0 * (w_kg / h_cm) ** 2
    return 0.5 * (_lbm_james128(w_kg, h_cm, "M") + _lbm_james128(w_kg, h_cm, "F"))

def _lbm_morgan(w_kg, h_cm, sex):
    if sex == "M":
        return 1.10 * w_kg - 120.0 * (w_kg / h_cm) ** 2
    if sex == "F":
        return 1.07 * w_kg - 148.0 * (w_kg / h_cm) ** 2
    return 0.5 * (_lbm_morgan(w_kg, h_cm, "M") + _lbm_morgan(w_kg, h_cm, "F"))

def _lbm_janmahasatian(w_kg, h_cm, sex):
    bmi = w_kg / (h_cm / 100.0) ** 2
    if sex == "M":
        return 9270.0 * w_kg / (6680.0 + 216.0 * bmi)
    if sex == "F":
        return 9270.0 * w_kg / (8780.0 + 244.0 * bmi)
    return 0.5 * (_lbm_janmahasatian(w_kg, h_cm, "M") + _lbm_janmahasatian(w_kg, h_cm, "F"))

def _ibw(h_cm, sex):
    if sex == "M":
        return 48.0 + 1.06 * (h_cm - 152.0)
    if sex == "F":
        return 45.5 + 0.91 * (h_cm - 152.0)
    return 0.5 * (_ibw(h_cm, "M") + _ibw(h_cm, "F"))  # FIX BUG-13

def _bsa_dubois_cm2(w_kg, h_cm):
    """Du Bois BSA in cm² (formula gives m², × 10000)."""
    return 0.007184 * (h_cm ** 0.725) * (w_kg ** 0.425) * 10000.0


# ──────────────────────────────────────────────────────────────────────────────
# CORE: compute SUV conversion factor for one slice
# ──────────────────────────────────────────────────────────────────────────────

def compute_suv_factor_for_slice(dcm, first_dcm) -> tuple:
    """
    Returns (suv_factor, label) such that:
        suv_voxel = (m * pixel + b) * suv_factor
    where m/b are the rescale slope/intercept for this slice.
    """
    radiopharm_seq = dcm[0x0054, 0x0016][0]

    # ── Patient weight (always kg, convert g if > 1000) ──────────────────────
    w_kg = float(dcm[0x0010, 0x1030].value)
    if w_kg > 1000:
        w_kg /= 1000.0
    w_g = w_kg * 1000.0

    # ── Half-life ─────────────────────────────────────────────────────────────
    half_life_s = float(radiopharm_seq[0x0018, 0x1075].value)
    lam = math.log(2) / half_life_s

    # ── Administered dose  FIX BUG-5 ─────────────────────────────────────────
    dose_raw = float(radiopharm_seq[0x0018, 0x1074].value)
    dose_bq  = dose_raw * 1e6 if dose_raw < 1e4 else dose_raw

    # ── Administration time  FIX BUG-10/11 ───────────────────────────────────
    series_date = str(dcm[0x0008, 0x0021].value).strip()
    admin_dt    = _get_administration_datetime(radiopharm_seq, series_date)

    # ── Units ─────────────────────────────────────────────────────────────────
    units_tag = str(dcm[0x0054, 0x1001].value).strip().upper() \
        if (0x0054, 0x1001) in dcm else "BQML"

    # ── Decay correction mode ─────────────────────────────────────────────────
    dc_mode = str(getattr(dcm, "DecayCorrection", "START")).strip().upper()

    # ══════════════════════════════════════════════════════════════════════════
    # FIX BUG-2/3/4: non-BQML units
    # ══════════════════════════════════════════════════════════════════════════

    if units_tag == "GML":
        # Values are already in SUV space; rescale slope still must be applied
        suv_type = str(dcm[0x0054, 0x1006].value).strip().upper() \
            if (0x0054, 0x1006) in dcm else ""

        if suv_type in ("", "BW"):
            return 1.0, "GML/BW"

        h_cm = float(dcm[0x0010, 0x1020].value) * 100.0
        sex  = str(getattr(dcm, "PatientSex", "O")).strip().upper()
        if sex not in ("M", "F"):
            sex = "O"

        if suv_type == "LBM":
            return w_g / (_lbm_morgan(w_kg, h_cm, sex) * 1000.0),    "GML/LBM→BW"
        if suv_type == "LBMJAMES128":
            return w_g / (_lbm_james128(w_kg, h_cm, sex) * 1000.0),  "GML/LBMJAMES128→BW"
        if suv_type == "LBMJANMA":
            return w_g / (_lbm_janmahasatian(w_kg, h_cm, sex) * 1000.0), "GML/LBMJANMA→BW"
        if suv_type == "IBW":
            return w_g / (_ibw(h_cm, sex) * 1000.0),                 "GML/IBW→BW"
        return 1.0, f"GML/{suv_type}(unknown→BW)"

    if units_tag == "CM2ML":
        # FIX BUG-3: BSA normalization → back-convert to SUVbw
        h_cm   = float(dcm[0x0010, 0x1020].value) * 100.0
        bsa_cm2 = _bsa_dubois_cm2(w_kg, h_cm)
        # stored value = Ac * BSA_cm2 / (D * 1e4)
        # SUVbw        = Ac * W_g / D
        # → factor = W_g / bsa_cm2
        return w_g / bsa_cm2, "CM2ML/BSA→BW"

    if units_tag == "CNTS":
        # FIX BUG-4 / FIX-15: Philips scale factors.
        # Convert CNTS → physical units, then fall through to the appropriate
        # DC correction path. This correctly handles the combined case where
        # Units=CNTS AND DC=NONE (DRO_3_4_2).
        if (0x7053, 0x1000) in dcm:
            suv_sf = float(dcm[0x7053, 0x1000].value)
            if suv_sf != 0.0:
                # SUV Scale Factor converts directly to SUVbw — no further
                # decay correction needed (already baked in by the scanner).
                return suv_sf, "CNTS/SUV_SF"

        if (0x7053, 0x1009) in dcm:
            ac_sf = float(dcm[0x7053, 0x1009].value)
            if ac_sf != 0.0:
                # Activity Concentration Scale Factor converts CNTS → Bq/mL.
                # Multiply pixel by ac_sf to get Bq/mL, then apply the normal
                # decay-corrected SUVbw formula for this slice's DC mode.
                # We do this by computing the BQML factor and scaling by ac_sf.
                if dc_mode == "ADMIN":
                    return (w_g / dose_bq) * ac_sf, "CNTS/AC_SF+DC=ADMIN"

                if dc_mode == "NONE":
                    # Per-frame correction: CNTS → Bq/mL → SUVbw per frame
                    acq_date = str(dcm[0x0008, 0x0022].value).strip()
                    acq_time = str(dcm[0x0008, 0x0032].value).strip().split(".")[0]
                    tacq     = _parse_dicom_datetime(acq_date, acq_time)
                    T_s      = float(dcm[0x0018, 0x1242].value) / 1000.0
                    adm_dt   = _midnight_corrected(admin_dt, tacq)
                    dt_s     = (tacq - adm_dt).total_seconds()
                    frame_factor = (lam * T_s / (1.0 - math.exp(-lam * T_s))) * math.exp(lam * dt_s)
                    return (w_g / dose_bq) * frame_factor * ac_sf, "CNTS/AC_SF+DC=NONE"

                # DC = START (default)
                ref_dt   = _get_reference_time_start(first_dcm, half_life_s)
                adm_dt   = _midnight_corrected(admin_dt, ref_dt)
                delta_s  = (ref_dt - adm_dt).total_seconds()
                dose_dec = dose_bq * math.exp(-lam * delta_s)
                return (w_g / dose_dec) * ac_sf, "CNTS/AC_SF+DC=START"

        raise ValueError("Units=CNTS but no usable Philips scale factor found.")

    # ══════════════════════════════════════════════════════════════════════════
    # BQML path
    # ══════════════════════════════════════════════════════════════════════════

    if dc_mode == "ADMIN":
        # FIX BUG-6: dose already at administration time, no decay correction
        return w_g / dose_bq, "BQML/DC=ADMIN"

    if dc_mode == "NONE":
        # FIX BUG-7: per-frame correction
        acq_date = str(dcm[0x0008, 0x0022].value).strip()
        acq_time = str(dcm[0x0008, 0x0032].value).strip().split(".")[0]
        tacq     = _parse_dicom_datetime(acq_date, acq_time)
        T_s      = float(dcm[0x0018, 0x1242].value) / 1000.0
        admin_dt = _midnight_corrected(admin_dt, tacq)
        dt_acq_adm = (tacq - admin_dt).total_seconds()
        frame_factor = (lam * T_s / (1.0 - math.exp(-lam * T_s))) * math.exp(lam * dt_acq_adm)
        return (w_g / dose_bq) * frame_factor, "BQML/DC=NONE"

    # DC = START (default)  FIX BUG-8
    ref_dt   = _get_reference_time_start(first_dcm, half_life_s)
    admin_dt = _midnight_corrected(admin_dt, ref_dt)
    delta_s  = (ref_dt - admin_dt).total_seconds()
    dose_dec = dose_bq * math.exp(-lam * delta_s)
    return w_g / dose_dec, "BQML/DC=START"


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: PET DICOM → SUV NIfTI  (FIX BUG-1: per-slice rescale slope)
# ──────────────────────────────────────────────────────────────────────────────

def dicom_pet_to_suv_nifti(series_dir: str, output_path: str) -> str:
    reader = sitk.ImageSeriesReader()
    dicom_names = sorted(reader.GetGDCMSeriesFileNames(series_dir))
    if not dicom_names:
        raise FileNotFoundError(f"No DICOM series in: {series_dir}")

    first_dcm = pydicom.dcmread(dicom_names[0])
    slices_suv = []
    label = None

    for fname in dicom_names:
        dcm = pydicom.dcmread(fname)

        # FIX BUG-1: per-slice rescale
        m = float(getattr(dcm, "RescaleSlope",     1.0))
        b = float(getattr(dcm, "RescaleIntercept", 0.0))

        pixels    = dcm.pixel_array.astype(np.float32)
        physical  = m * pixels + b        

        suv_factor, lbl = compute_suv_factor_for_slice(dcm, first_dcm)
        if label is None:
            label = lbl

        slices_suv.append(physical * suv_factor)

    print(f"    Conversion mode: {label}")
    print(f"    SUV factor (s0): {suv_factor:.6g}")

    volume = np.stack(slices_suv, axis=0).astype(np.float32)

    # Copy geometry from SimpleITK reader
    geo_reader = sitk.ImageSeriesReader()
    geo_reader.SetFileNames(dicom_names)
    ref_img = geo_reader.Execute()

    suv_img = sitk.GetImageFromArray(volume)
    suv_img.CopyInformation(ref_img)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(suv_img, output_path)
    print(f"    Saved: {output_path}")
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: mask handling
# ──────────────────────────────────────────────────────────────────────────────

def rtstruct_to_nifti_mask(series_dir: str, rtstruct_path: str, output_path: str):
    try:
        from rt_utils import RTStructBuilder
    except ImportError:
        sys.exit("[ERROR] rt-utils not installed. Run: pip install rt-utils")
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=series_dir, rt_struct_path=rtstruct_path
    )
    roi_names = rtstruct.get_roi_names()
    print(f"    ROIs: {roi_names}")
    masks    = [rtstruct.get_roi_mask_by_name(r).astype(np.uint8) for r in roi_names]
    combined = np.where(sum(masks) >= 1, 1, 0).astype(np.uint8)
    combined = np.moveaxis(combined, [0, 1, 2], [1, 2, 0])
    reader   = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(series_dir))
    ref_img  = reader.Execute()
    mask_img = sitk.GetImageFromArray(combined)
    mask_img.CopyInformation(ref_img)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(mask_img, output_path)
    print(f"    Mask saved: {output_path}")


def copy_nifti_mask(src: str, dst: str):
    import shutil
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"    Mask copied: {dst}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: statistics inside mask
# ──────────────────────────────────────────────────────────────────────────────

def extract_suv_stats(suv_path: str, mask_path: str) -> dict:
    suv_data  = nib.load(suv_path).get_fdata(dtype=np.float32)
    mask_data = nib.load(mask_path).get_fdata() >= 0.5
    vox = suv_data[mask_data]
    if vox.size == 0:
        return {"mean": np.nan, "max": np.nan, "min": np.nan,
                "median": np.nan, "std": np.nan, "n_voxels": 0}
    return {
        "mean":     float(np.mean(vox)),
        "max":      float(np.max(vox)),
        "min":      float(np.min(vox)),
        "median":   float(np.median(vox)),
        "std":      float(np.std(vox)),
        "n_voxels": int(vox.size),
    }


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: discover DRO series
# ──────────────────────────────────────────────────────────────────────────────

def find_dro_series(dro_root: str) -> list:
    series_list = []
    for dro_name in sorted(os.listdir(dro_root)):
        dro_dir = os.path.join(dro_root, dro_name)
        if not os.path.isdir(dro_dir):
            continue
        dicom_dir = rtstruct_path = nifti_mask_path = None
        for root, dirs, files in os.walk(dro_dir):
            dcm_files = [f for f in files if f.lower().endswith(".dcm")]
            if not dcm_files:
                continue
            sample = pydicom.dcmread(
                os.path.join(root, dcm_files[0]),
                stop_before_pixels=True, force=True,
            )
            modality = getattr(sample, "Modality", "")
            if modality == "PT" and dicom_dir is None:
                dicom_dir = root
            elif modality == "RTSTRUCT" and rtstruct_path is None:
                rtstruct_path = os.path.join(root, dcm_files[0])
        mask_dir = os.path.join(dro_dir, "mask")
        if os.path.isdir(mask_dir):
            for f in os.listdir(mask_dir):
                if f.lower().endswith((".nii", ".nii.gz")):
                    nifti_mask_path = os.path.join(mask_dir, f)
                    break
        if dicom_dir is None:
            print(f"[SKIP] {dro_name}: no PET DICOM series found.")
            continue
        series_list.append({
            "name": dro_name, "dicom_dir": dicom_dir,
            "rtstruct_path": rtstruct_path, "nifti_mask_path": nifti_mask_path,
        })
    return series_list


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

DRO_EXPECTED = {"min": 0.20, "median": 1.00, "max": 4.00}


def ensure_repo():
    if os.path.isdir(REPO_DIR):
        print(f"[INFO] Repository present at: {REPO_DIR}")
        return
    print(f"[INFO] Cloning {REPO_URL} …")
    r = subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, REPO_DIR],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit(f"[ERROR] git clone failed:\n{r.stderr}")
    print("[INFO] Clone complete.")


def run_pipeline():
    ensure_repo()
    if not os.path.isdir(DRO_ROOT):
        sys.exit(f"[ERROR] DRO directory not found: {DRO_ROOT}")

    output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "suv_output")
    os.makedirs(output_root, exist_ok=True)

    series_list = find_dro_series(DRO_ROOT)
    if not series_list:
        sys.exit("[ERROR] No PET DICOM series found under DRO/.")

    print(f"\n[INFO] Found {len(series_list)} DRO series.\n")
    results = []

    for entry in series_list:
        name, dicom_dir = entry["name"], entry["dicom_dir"]
        rtstruct, nifti_mask = entry["rtstruct_path"], entry["nifti_mask_path"]

        print(f"── Processing {name} ──────────────────────────────")
        out_suv  = os.path.join(output_root, name, f"{name}_suv.nii.gz")
        out_mask = os.path.join(output_root, name, f"{name}_mask.nii.gz")

        try:
            dicom_pet_to_suv_nifti(dicom_dir, out_suv)
        except Exception as e:
            print(f"    [ERROR] SUV conversion failed: {e}")
            results.append({"DRO": name, "error": str(e)})
            print()
            continue

        mask_ready = False
        if nifti_mask and os.path.isfile(nifti_mask):
            copy_nifti_mask(nifti_mask, out_mask)
            mask_ready = True
        elif rtstruct and os.path.isfile(rtstruct):
            try:
                rtstruct_to_nifti_mask(dicom_dir, rtstruct, out_mask)
                mask_ready = True
            except Exception as e:
                print(f"    [ERROR] RTSTRUCT→mask failed: {e}")

        if mask_ready:
            try:
                s = extract_suv_stats(out_suv, out_mask)
                ok = (abs(s["min"]    - DRO_EXPECTED["min"])    < 0.01 and
                      abs(s["median"] - DRO_EXPECTED["median"]) < 0.01 and
                      abs(s["max"]    - DRO_EXPECTED["max"])    < 0.01)
                status = "✓ PASS" if ok else "✗ FAIL"
                results.append({"DRO": name, "status": status, **s})
                print(f"    min={s['min']:.4f}  median={s['median']:.4f}  "
                      f"max={s['max']:.4f}  → {status}")
            except Exception as e:
                print(f"    [ERROR] Stats failed: {e}")
                results.append({"DRO": name, "error": str(e)})
        print()

    exp = DRO_EXPECTED
    print("=" * 80)
    print(f"  Expected: min={exp['min']}  median={exp['median']}  max={exp['max']}")
    print("-" * 80)
    print(f"  {'DRO':<15} {'Status':<10} {'min':>8} {'median':>8} {'max':>8}  "
          f"{'Δmin':>8} {'Δmed':>8} {'Δmax':>8}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"  {r['DRO']:<15} ERROR      {r['error'][:45]}")
        else:
            dm, dd, dx = r["min"]-exp["min"], r["median"]-exp["median"], r["max"]-exp["max"]
            print(f"  {r['DRO']:<15} {r['status']:<10} "
                  f"{r['min']:>8.4f} {r['median']:>8.4f} {r['max']:>8.4f}  "
                  f"{dm:>+8.4f} {dd:>+8.4f} {dx:>+8.4f}")
    print("=" * 80)
    print(f"\nOutputs: {output_root}")


if __name__ == "__main__":
    run_pipeline()