"""
Per-lesion PET/CT metrics for the mini lesion captioning pipeline.

Consumes the outputs already produced by closest_anatomy.py (ct.nii.gz,
segmentation_<task>.nii.gz, closest_anatomy.csv) plus the patient's PET DICOM
series, and computes, for every lesion:

  - closest_anatomy         (from closest_anatomy.csv)
  - lesion_centroid_mm      (x, y, z), patient LPS space
  - closest_anatomy centroid_mm (x, y, z), same space
  - suv_max / suv_mean_whole
  - axial_slice_index       (CT slice the lesion centroid falls on)
  - volume_mm3              (whole ROI mask volume, native PET grid)
  - mtv_ml                  (metabolic tumor volume: PET voxels inside the
                              ROI at or above 41% of that lesion's SUVmax)
  - tlg_g                   (total lesion glycolysis = mtv_ml * suv_mean_active)

Usage:
    python scripts/lesion_metrics.py --patient-dir data/11-37493
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from rt_utils import RTStructBuilder, image_helper

from totalsegmentator.map_to_binary import class_map

from closest_anatomy import (
    find_patient_series_dirs,
    pick_rtstruct_reference_series,
    get_lesion_roi_names,
    sitk_affine,
)

MTV_THRESHOLD_FRACTION = 0.41  # standard 41%-of-SUVmax MTV threshold


def dicom_time_to_seconds(time_str: str) -> float:
    time_str = str(time_str).split(".")[0]
    return int(time_str[0:2]) * 3600 + int(time_str[2:4]) * 60 + int(time_str[4:6])


def build_suv_volume(series_data) -> np.ndarray:
    """SUVbw volume in the same (Columns, Rows, slices) = (x, y, z) layout rt-utils masks use."""
    first = series_data[0]
    assert first.Units == "BQML", f"Unexpected PET Units: {first.Units}"

    weight_g = float(first.PatientWeight) * 1000
    rp = first.RadiopharmaceuticalInformationSequence[0]
    injected_dose_bq = float(rp.RadionuclideTotalDose)
    half_life_s = float(rp.RadionuclideHalfLife)
    injection_s = dicom_time_to_seconds(rp.RadiopharmaceuticalStartTime)
    reference_s = dicom_time_to_seconds(first.SeriesTime)
    decay_s = reference_s - injection_s
    if decay_s < 0:
        decay_s += 24 * 3600
    decayed_dose_bq = injected_dose_bq * (0.5 ** (decay_s / half_life_s))

    slices = []
    for s in series_data:
        raw = s.pixel_array.T.astype(np.float64)  # -> (Columns, Rows) = (x, y)
        slope = float(getattr(s, "RescaleSlope", 1.0))
        intercept = float(getattr(s, "RescaleIntercept", 0.0))
        slices.append(raw * slope + intercept)  # Bq/mL
    activity_bqml = np.stack(slices, axis=-1)  # (x, y, z)

    return activity_bqml * weight_g / decayed_dose_bq


def pet_space_to_sitk_image(array_xyz: np.ndarray, pixel_to_patient: np.ndarray) -> sitk.Image:
    """Wrap a (x, y, z) array (rt-utils layout) into a geometrically correct sitk.Image."""
    origin = pixel_to_patient[:3, 3].astype(np.float64)
    cols = [pixel_to_patient[:3, i].astype(np.float64) for i in range(3)]
    spacing = tuple(float(np.linalg.norm(c)) for c in cols)
    direction = np.column_stack([c / s for c, s in zip(cols, spacing)])

    array_numeric = array_xyz.astype(np.uint8) if array_xyz.dtype == bool else array_xyz
    img = sitk.GetImageFromArray(np.transpose(array_numeric, (2, 1, 0)))
    img.SetOrigin(tuple(float(v) for v in origin))
    img.SetSpacing(spacing)
    img.SetDirection(tuple(float(v) for v in direction.flatten()))
    return img


def resample_to_ct(array_xyz: np.ndarray, pixel_to_patient: np.ndarray, ct_image: sitk.Image, is_mask: bool) -> np.ndarray:
    src_img = pet_space_to_sitk_image(array_xyz, pixel_to_patient)
    if is_mask:
        src_img = sitk.Cast(src_img, sitk.sitkUInt8)
        interpolator, out_type, default = sitk.sitkNearestNeighbor, sitk.sitkUInt8, 0
    else:
        src_img = sitk.Cast(src_img, sitk.sitkFloat32)
        interpolator, out_type, default = sitk.sitkLinear, sitk.sitkFloat32, 0.0

    resampled = sitk.Resample(src_img, ct_image, sitk.Transform(), interpolator, default, out_type)
    return np.transpose(sitk.GetArrayFromImage(resampled), (2, 1, 0))


def voxel_to_physical(verts_ijk: np.ndarray, affine: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([verts_ijk, np.ones((len(verts_ijk), 1))], axis=1)
    return (affine @ homogeneous.T).T[:, :3]


class PatientContext:
    """Everything loaded once per patient: images, segmentation, RTSTRUCT, PET/SUV."""

    def __init__(self, patient_dir: Path, output_dir: Path | None = None, task: str = "total"):
        self.patient_dir = patient_dir
        self.task = task
        self.output_dir = output_dir or Path("outputs") / patient_dir.name

        ct_nifti_path = self.output_dir / "ct.nii.gz"
        seg_nifti_path = self.output_dir / f"segmentation_{task}.nii.gz"
        csv_path = self.output_dir / "closest_anatomy.csv"
        for p in (ct_nifti_path, seg_nifti_path, csv_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found. Run scripts/closest_anatomy.py --patient-dir {patient_dir} first."
                )

        self.ct_dir, self.pt_dir, self.rtstruct_file = find_patient_series_dirs(patient_dir)
        if self.pt_dir is None:
            raise FileNotFoundError(f"No PET series found under {patient_dir}")
        self.reference_series_dir = pick_rtstruct_reference_series(self.rtstruct_file, self.ct_dir, self.pt_dir)

        self.ct_image = sitk.ReadImage(str(ct_nifti_path))
        self.seg_image = sitk.ReadImage(str(seg_nifti_path))
        self.ct_arr = np.transpose(sitk.GetArrayFromImage(self.ct_image), (2, 1, 0))
        self.seg_arr = np.transpose(sitk.GetArrayFromImage(self.seg_image), (2, 1, 0))
        self.ct_affine = sitk_affine(self.ct_image)
        self.ct_affine_inv = np.linalg.inv(self.ct_affine)
        self.ct_spacing = self.ct_image.GetSpacing()
        self.ct_size = self.ct_image.GetSize()

        self.class_names = class_map[task]
        self.name_to_label = {name: label for label, name in self.class_names.items()}

        self.closest_df = pd.read_csv(csv_path).set_index("lesion")

        self.rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(self.reference_series_dir), rt_struct_path=str(self.rtstruct_file)
        )
        self.pixel_to_patient = image_helper.get_pixel_to_patient_transformation_matrix(self.rtstruct.series_data)
        self.lesion_names = get_lesion_roi_names(self.rtstruct, r"suv\s*peak\s*sphere")

        self.pet_voxel_volume_mm3 = (
            np.linalg.norm(self.pixel_to_patient[:3, 0])
            * np.linalg.norm(self.pixel_to_patient[:3, 1])
            * np.linalg.norm(self.pixel_to_patient[:3, 2])
        )

        self.suv_volume = build_suv_volume(self.rtstruct.series_data)

        self._lesion_masks_pet: dict[str, np.ndarray] = {}
        self._lesion_masks_ct: dict[str, np.ndarray] = {}
        self._suv_ct: np.ndarray | None = None

    def lesion_mask_pet(self, name: str) -> np.ndarray:
        if name not in self._lesion_masks_pet:
            self._lesion_masks_pet[name] = self.rtstruct.get_roi_mask_by_name(name)
        return self._lesion_masks_pet[name]

    def lesion_mask_ct(self, name: str) -> np.ndarray:
        if name not in self._lesion_masks_ct:
            self._lesion_masks_ct[name] = resample_to_ct(
                self.lesion_mask_pet(name), self.pixel_to_patient, self.ct_image, is_mask=True
            ).astype(bool)
        return self._lesion_masks_ct[name]

    @property
    def suv_ct(self) -> np.ndarray:
        """SUV volume resampled onto the CT grid, for display only (metrics use native PET grid)."""
        if self._suv_ct is None:
            self._suv_ct = resample_to_ct(self.suv_volume, self.pixel_to_patient, self.ct_image, is_mask=False)
        return self._suv_ct

    def organ_centroid_mm(self, organ_name: str) -> np.ndarray | None:
        label = self.name_to_label.get(organ_name)
        if label is None:
            return None
        coords = np.argwhere(self.seg_arr == label)
        if len(coords) == 0:
            return None
        return voxel_to_physical(coords.mean(axis=0, keepdims=True), self.ct_affine)[0]

    def ct_index_of_point(self, point_mm) -> np.ndarray:
        homogeneous = np.append(np.asarray(point_mm, dtype=float), 1.0)
        return (self.ct_affine_inv @ homogeneous)[:3]


def compute_lesion_report(ctx: PatientContext, mtv_threshold_fraction: float = MTV_THRESHOLD_FRACTION) -> pd.DataFrame:
    rows = []
    for name in ctx.lesion_names:
        mask = ctx.lesion_mask_pet(name)
        suv_vals = ctx.suv_volume[mask]

        closest_row = ctx.closest_df.loc[name] if name in ctx.closest_df.index else None
        closest_anatomy = closest_row["closest_anatomy"] if closest_row is not None else None
        distance_mm = closest_row["distance_mm"] if closest_row is not None else None
        lesion_centroid_mm = (
            np.array(ast.literal_eval(closest_row["centroid_xyz_mm"]))
            if closest_row is not None
            else image_helper.apply_transformation_to_3d_points(
                np.argwhere(mask).astype(float), ctx.pixel_to_patient
            ).mean(axis=0)
        )

        if len(suv_vals) == 0:
            rows.append({"lesion": name, "closest_anatomy": closest_anatomy, "distance_mm": distance_mm,
                         "lesion_centroid_x_mm": lesion_centroid_mm[0], "lesion_centroid_y_mm": lesion_centroid_mm[1],
                         "lesion_centroid_z_mm": lesion_centroid_mm[2]})
            continue

        suv_max = float(suv_vals.max())
        suv_mean_whole = float(suv_vals.mean())
        volume_mm3 = float(mask.sum()) * ctx.pet_voxel_volume_mm3

        threshold = mtv_threshold_fraction * suv_max
        active = mask & (ctx.suv_volume >= threshold)
        mtv_ml = float(active.sum()) * ctx.pet_voxel_volume_mm3 / 1000.0
        suv_mean_active = float(ctx.suv_volume[active].mean()) if active.any() else suv_max
        tlg_g = mtv_ml * suv_mean_active

        organ_centroid = ctx.organ_centroid_mm(closest_anatomy) if closest_anatomy else None
        axial_idx = int(round(ctx.ct_index_of_point(lesion_centroid_mm)[2]))
        axial_idx = max(0, min(ctx.ct_size[2] - 1, axial_idx))

        rows.append({
            "lesion": name,
            "closest_anatomy": closest_anatomy,
            "distance_mm": distance_mm,
            "lesion_centroid_x_mm": lesion_centroid_mm[0],
            "lesion_centroid_y_mm": lesion_centroid_mm[1],
            "lesion_centroid_z_mm": lesion_centroid_mm[2],
            "closest_anatomy_centroid_x_mm": organ_centroid[0] if organ_centroid is not None else None,
            "closest_anatomy_centroid_y_mm": organ_centroid[1] if organ_centroid is not None else None,
            "closest_anatomy_centroid_z_mm": organ_centroid[2] if organ_centroid is not None else None,
            "axial_slice_index": axial_idx,
            "suv_max": suv_max,
            "suv_mean_whole": suv_mean_whole,
            "volume_mm3": volume_mm3,
            "mtv_ml": mtv_ml,
            "suv_mean_active": suv_mean_active,
            "tlg_g": tlg_g,
            "n_voxels_pet": int(mask.sum()),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patient-dir", type=Path, default=Path("data/11-37493"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to outputs/<patient-dir name>")
    parser.add_argument("--task", default="total")
    parser.add_argument("--mtv-threshold-fraction", type=float, default=MTV_THRESHOLD_FRACTION)
    parser.add_argument("--csv-out", type=Path, default=None, help="Defaults to <output-dir>/lesion_report.csv")
    args = parser.parse_args()

    ctx = PatientContext(args.patient_dir, args.output_dir, args.task)
    report = compute_lesion_report(ctx, args.mtv_threshold_fraction)

    csv_out = args.csv_out or ctx.output_dir / "lesion_report.csv"
    report.to_csv(csv_out, index=False)
    print(f"Wrote {len(report)} lesion rows to {csv_out}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
