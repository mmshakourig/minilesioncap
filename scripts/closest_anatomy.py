"""
For each lesion contoured in a patient's RT-STRUCT, find the closest anatomical
structure segmented by TotalSegmentator (https://github.com/wasserth/TotalSegmentator)
in the patient's CT.

Pipeline:
  1. Discover the patient's CT, PET and RTSTRUCT DICOM files under --patient-dir.
  2. Convert the CT series to NIfTI (SimpleITK) and run TotalSegmentator on it to get
     a multi-label organ segmentation in CT voxel space.
  3. Load the RTSTRUCT with rt-utils, using whichever series (CT or PET) it was
     actually contoured against (determined from the RTSTRUCT's referenced
     SeriesInstanceUID, not assumed).
  4. For every lesion ROI, map its contour voxels into CT physical space and then
     into CT voxel indices (both series share the same DICOM FrameOfReferenceUID).
  5. Look up the nearest non-background organ label for those voxels (0mm if the
     lesion overlaps a segmented organ, otherwise the Euclidean distance in mm to
     the nearest one) and report it as the lesion's closest anatomy.

Usage:
    python scripts/closest_anatomy.py --patient-dir data/11-37493
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from rt_utils import RTStructBuilder, image_helper
from scipy.ndimage import distance_transform_edt

from totalsegmentator.map_to_binary import class_map
from totalsegmentator.python_api import totalsegmentator


def find_patient_series_dirs(patient_dir: Path):
    """Locate the CT, PET and RTSTRUCT DICOM folders for a patient export."""
    ct_dir = pt_dir = None
    rtstruct_file = None

    for child in sorted(patient_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name.upper()
        if "_CT_" in name:
            ct_dir = child
        elif "_PT_" in name or "_PET" in name:
            pt_dir = child
        elif "_RTST" in name or "_RTSTRUCT" in name:
            dcm_files = sorted(child.glob("*.dcm"))
            if not dcm_files:
                raise FileNotFoundError(f"No .dcm file found in RTSTRUCT folder {child}")
            if len(dcm_files) > 1:
                raise RuntimeError(
                    f"Expected exactly one RTSTRUCT file in {child}, found {len(dcm_files)}"
                )
            rtstruct_file = dcm_files[0]

    if ct_dir is None:
        raise FileNotFoundError(f"Could not find a CT series folder under {patient_dir}")
    if rtstruct_file is None:
        raise FileNotFoundError(f"Could not find an RTSTRUCT folder under {patient_dir}")

    return ct_dir, pt_dir, rtstruct_file


def series_instance_uid(dicom_dir: Path) -> str:
    first_file = next(dicom_dir.glob("*.dcm"))
    ds = pydicom.dcmread(first_file, stop_before_pixels=True)
    return ds.SeriesInstanceUID


def pick_rtstruct_reference_series(rtstruct_file: Path, ct_dir: Path, pt_dir: Path | None) -> Path:
    """Return whichever series folder (CT or PET) the RTSTRUCT contours were drawn on."""
    ds = pydicom.dcmread(rtstruct_file, stop_before_pixels=True)
    referenced_uids = set()
    for frame_of_ref in ds.ReferencedFrameOfReferenceSequence:
        for study in frame_of_ref.RTReferencedStudySequence:
            for series in study.RTReferencedSeriesSequence:
                referenced_uids.add(series.SeriesInstanceUID)

    ct_uid = series_instance_uid(ct_dir)
    if ct_uid in referenced_uids:
        return ct_dir

    if pt_dir is not None:
        pt_uid = series_instance_uid(pt_dir)
        if pt_uid in referenced_uids:
            return pt_dir

    raise RuntimeError(
        "RTSTRUCT does not reference the CT series or the PET series found for this patient; "
        f"referenced SeriesInstanceUID(s): {referenced_uids}"
    )


def convert_ct_to_nifti(ct_dir: Path, out_path: Path) -> sitk.Image:
    if out_path.exists():
        return sitk.ReadImage(str(out_path))

    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(str(ct_dir))
    reader.SetFileNames(dicom_files)
    ct_image = reader.Execute()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(ct_image, str(out_path))
    return ct_image


def run_total_segmentator(ct_nifti_path: Path, out_path: Path, task: str, fast: bool, device: str):
    if not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        totalsegmentator(
            input=str(ct_nifti_path),
            output=str(out_path),
            task=task,
            ml=True,
            fast=fast,
            device=device,
            quiet=True,
        )
    return sitk.ReadImage(str(out_path))


def sitk_affine(image: sitk.Image) -> np.ndarray:
    """Voxel index (x, y, z) -> physical point (mm), same LPS space DICOM uses."""
    direction = np.array(image.GetDirection()).reshape(3, 3)
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())

    affine = np.eye(4)
    affine[:3, :3] = direction * spacing  # column i scaled by spacing[i]
    affine[:3, 3] = origin
    return affine


def physical_points_to_indices(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Vectorized physical (mm) -> continuous voxel index (x, y, z), given a sitk-style affine."""
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (np.linalg.inv(affine) @ homogeneous.T).T[:, :3]


def build_nearest_anatomy_maps(seg_xyz: np.ndarray, spacing_xyz: tuple):
    """
    For every voxel, find the label of the nearest non-background structure and its
    distance in mm (0 mm for voxels already inside a structure).
    """
    background = seg_xyz == 0
    distances, nearest_idx = distance_transform_edt(
        background, sampling=spacing_xyz, return_indices=True
    )

    nearest_label = seg_xyz.copy()
    nearest_label[background] = seg_xyz[tuple(nearest_idx[:, background])]
    distances[~background] = 0.0
    return nearest_label, distances


def get_lesion_roi_names(rtstruct, exclude_pattern: str):
    pattern = re.compile(exclude_pattern, re.IGNORECASE)
    return [name for name in rtstruct.get_roi_names() if not pattern.search(name)]


def closest_anatomy_for_lesion(
    mask: np.ndarray,
    pixel_to_patient: np.ndarray,
    ct_affine: np.ndarray,
    ct_shape_xyz: tuple,
    nearest_label: np.ndarray,
    nearest_distance: np.ndarray,
    class_names: dict,
):
    voxel_idx = np.argwhere(mask).astype(float)  # (col, row, slice) in the RTSTRUCT series
    physical_pts = image_helper.apply_transformation_to_3d_points(voxel_idx, pixel_to_patient)

    ct_idx = physical_points_to_indices(physical_pts, ct_affine)
    ct_idx_round = np.rint(ct_idx).astype(int)

    in_bounds = np.all((ct_idx_round >= 0) & (ct_idx_round < np.array(ct_shape_xyz)), axis=1)
    n_dropped = int((~in_bounds).sum())
    ct_idx_round = ct_idx_round[in_bounds]

    if len(ct_idx_round) == 0:
        return {
            "closest_anatomy": None,
            "distance_mm": None,
            "overlap_fraction": 0.0,
            "n_voxels": int(mask.sum()),
            "n_voxels_out_of_ct_fov": n_dropped,
            "centroid_xyz_mm": physical_pts.mean(axis=0).tolist(),
        }

    ix, iy, iz = ct_idx_round[:, 0], ct_idx_round[:, 1], ct_idx_round[:, 2]
    voxel_labels = nearest_label[ix, iy, iz]
    voxel_distances = nearest_distance[ix, iy, iz]

    min_distance = voxel_distances.min()
    winners = voxel_labels[voxel_distances == min_distance]
    winner_label = np.bincount(winners).argmax()

    overlap_fraction = float((voxel_labels == winner_label).mean())

    return {
        "closest_anatomy": class_names.get(int(winner_label), f"label_{winner_label}"),
        "distance_mm": float(min_distance),
        "overlap_fraction": overlap_fraction,
        "n_voxels": int(mask.sum()),
        "n_voxels_out_of_ct_fov": n_dropped,
        "centroid_xyz_mm": physical_pts.mean(axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patient-dir", type=Path, default=Path("data/11-37493"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to outputs/<patient-dir name>")
    parser.add_argument("--task", default="total", help="TotalSegmentator task (default: total)")
    parser.add_argument("--fast", action="store_true", default=True, help="Use TotalSegmentator's fast (3mm) model")
    parser.add_argument("--full-res", dest="fast", action="store_false", help="Use the full-resolution model instead of --fast")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument(
        "--exclude-roi-pattern",
        default=r"suv\s*peak\s*sphere",
        help="Regex for ROI names to exclude from the lesion set (default excludes SUV peak sphere markers)",
    )
    parser.add_argument("--csv-out", type=Path, default=None, help="Defaults to <output-dir>/closest_anatomy.csv")
    args = parser.parse_args()

    if args.device == "auto":
        import torch
        args.device = "gpu" if torch.cuda.is_available() else "cpu"

    output_dir = args.output_dir or Path("outputs") / args.patient_dir.name
    csv_out = args.csv_out or output_dir / "closest_anatomy.csv"

    ct_dir, pt_dir, rtstruct_file = find_patient_series_dirs(args.patient_dir)
    print(f"CT series:        {ct_dir}")
    print(f"PET series:       {pt_dir}")
    print(f"RTSTRUCT file:    {rtstruct_file}")

    reference_series_dir = pick_rtstruct_reference_series(rtstruct_file, ct_dir, pt_dir)
    print(f"RTSTRUCT contoured against: {reference_series_dir}")

    ct_nifti_path = output_dir / "ct.nii.gz"
    ct_image = convert_ct_to_nifti(ct_dir, ct_nifti_path)
    print(f"CT NIfTI:         {ct_nifti_path} (size={ct_image.GetSize()})")

    seg_nifti_path = output_dir / f"segmentation_{args.task}.nii.gz"
    print(f"Running TotalSegmentator (task={args.task}, fast={args.fast}, device={args.device})...")
    seg_image = run_total_segmentator(ct_nifti_path, seg_nifti_path, args.task, args.fast, args.device)
    print(f"Segmentation:     {seg_nifti_path}")

    # SimpleITK numpy arrays come back as (z, y, x); transpose to (x, y, z) so voxel
    # indices line up 1:1 with GetSize()/spacing/affine order used everywhere else.
    seg_xyz = np.transpose(sitk.GetArrayFromImage(seg_image), (2, 1, 0))
    ct_affine = sitk_affine(ct_image)
    ct_shape_xyz = ct_image.GetSize()

    print("Building nearest-anatomy distance map...")
    nearest_label, nearest_distance = build_nearest_anatomy_maps(seg_xyz, ct_image.GetSpacing())

    class_names = class_map[args.task]

    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=str(reference_series_dir), rt_struct_path=str(rtstruct_file)
    )
    pixel_to_patient = image_helper.get_pixel_to_patient_transformation_matrix(rtstruct.series_data)

    lesion_names = get_lesion_roi_names(rtstruct, args.exclude_roi_pattern)
    print(f"Found {len(lesion_names)} lesion ROI(s) (excluded {len(rtstruct.get_roi_names()) - len(lesion_names)} marker ROI(s))")

    rows = []
    for name in lesion_names:
        mask = rtstruct.get_roi_mask_by_name(name)
        result = closest_anatomy_for_lesion(
            mask, pixel_to_patient, ct_affine, ct_shape_xyz, nearest_label, nearest_distance, class_names
        )
        rows.append({"lesion": name, **result})
        print(
            f"  {name}: closest_anatomy={result['closest_anatomy']}, "
            f"distance_mm={result['distance_mm']}, overlap_fraction={result['overlap_fraction']:.2f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    import csv

    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote results for {len(rows)} lesions to {csv_out}")


if __name__ == "__main__":
    main()
