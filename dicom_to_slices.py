"""
dicom_to_slices.py

Converts a folder of DICOM files (one or more series, including nested
subfolders) into:
  - a set of PNG files per series (one file per slice), normalized to
    8-bit for display
  - a manifest.json per series with geometry (order, Z positions, spacing),
    which Lens Studio / Spectacles uses to place the planes in the
    correct physical order.

Series are identified by the SeriesInstanceUID tag -- if a folder mixes
files from different series (different orientation, different studies),
each series is laid out into its own result subfolder.

Usage:
    python3 dicom_to_slices.py <input_dir> <output_dir> [--max-slices N]

Example:
    python3 dicom_to_slices.py ./my_scan ./output --max-slices 60

PROTOTYPE: --max-slices caps the result to the first N slices after
sorting by Z (from one physical edge of the volume), without decimation.
This is a deliberate performance limitation for the current version --
see the project documentation.
"""

import sys
import json
import os
import argparse
import numpy as np
import pydicom
from PIL import Image


def load_series(input_dir: str) -> list:
    """Reads all .dcm files from a folder (including subfolders) and
    returns them as pydicom datasets."""
    datasets = []
    for root, _dirs, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith(".dcm"):
                continue
            path = os.path.join(root, fname)
            try:
                ds = pydicom.dcmread(path)
                datasets.append(ds)
            except Exception as e:
                print(f"  [skipped] {fname}: {e}")

    if not datasets:
        raise RuntimeError(f"No valid .dcm files found in {input_dir}")

    return datasets


def group_by_series(datasets: list) -> dict:
    """Groups datasets by SeriesInstanceUID -- different series (different
    orientation, different studies, different modalities) must not be
    mixed into a single slice stack."""
    groups: dict = {}
    for ds in datasets:
        series_uid = str(getattr(ds, "SeriesInstanceUID", "UNKNOWN_SERIES"))
        groups.setdefault(series_uid, []).append(ds)
    return groups


def describe_series(series_uid: str, datasets: list) -> dict:
    """Builds a human-readable summary of a series -- used to print the
    series list before conversion."""
    first = datasets[0]
    total_frames = sum(int(getattr(ds, "NumberOfFrames", 1) or 1) for ds in datasets)
    return {
        "series_uid": series_uid,
        "file_count": len(datasets),
        "total_slices": total_frames,
        "modality": str(getattr(first, "Modality", "UNKNOWN")),
        "series_description": str(getattr(first, "SeriesDescription", "")).strip(),
        "rows": int(getattr(first, "Rows", 0)),
        "cols": int(getattr(first, "Columns", 0)),
    }


def frame_z_position(ds, frame_index: int, n_frames: int) -> float:
    """Returns the Z position of a specific frame inside a multi-frame file.

    Multi-frame DICOM stores per-frame geometry either in
    PerFrameFunctionalGroupsSequence (the correct path, but not always
    populated) or doesn't store it per frame at all -- in that case frames
    are spread out evenly along Z, using the file's base position plus the
    slice thickness step.
    """
    try:
        groups = ds.PerFrameFunctionalGroupsSequence
        frame_group = groups[frame_index]
        plane_pos = frame_group.PlanePositionSequence[0]
        return float(plane_pos.ImagePositionPatient[2])
    except (AttributeError, IndexError, KeyError):
        pass

    # fallback: spread frames evenly around the file's base position
    base_z = 0.0
    if hasattr(ds, "ImagePositionPatient"):
        base_z = float(ds.ImagePositionPatient[2])
    elif hasattr(ds, "SliceLocation"):
        base_z = float(ds.SliceLocation)

    step = float(
        getattr(ds, "SpacingBetweenSlices", None)
        or getattr(ds, "SliceThickness", 1.0)
    )
    # center frames around base_z so it stays the "average" position of the file
    offset = (frame_index - (n_frames - 1) / 2) * step
    return base_z + offset


def collect_frames(datasets: list) -> list:
    """Unfolds each pydicom dataset into a list of individual 'slices',
    accounting for multi-frame files (NumberOfFrames > 1). Each element is
    a dict with the ready 2D pixel_array, the dataset (for metadata), and
    the Z position.
    """
    frames = []
    for ds in datasets:
        n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        pixel_array = ds.pixel_array

        if n_frames == 1:
            # regular "one slice per file" file -- pixel_array is already 2D
            arr_2d = pixel_array
            z = frame_z_position(ds, 0, 1)
            frames.append({"ds": ds, "pixel_array": arr_2d, "z": z})
        else:
            # multi-frame file -- pixel_array has shape (n_frames, rows, cols)
            for f in range(n_frames):
                arr_2d = pixel_array[f]
                z = frame_z_position(ds, f, n_frames)
                frames.append({"ds": ds, "pixel_array": arr_2d, "z": z})

    return frames


def sort_frames_by_position(frames: list) -> list:
    """Sorts slices by physical position (Z), not by file order -- file/
    frame order in real-world exports is not guaranteed."""
    return sorted(frames, key=lambda f: f["z"])


def apply_windowing(pixel_array: np.ndarray, ds) -> np.ndarray:
    """Converts raw pixel values (HU for CT, arbitrary for MRI) into an
    8-bit 0-255 range for display, using the window center/width from the
    file itself (if present) or reasonable defaults based on the full range."""

    arr = pixel_array.astype(np.float32)

    # RescaleSlope/Intercept convert raw pixel values into physical units (HU for CT)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept

    window_center = getattr(ds, "WindowCenter", None)
    window_width = getattr(ds, "WindowWidth", None)

    # some files store several windowing presets as a list -- take the first one
    if isinstance(window_center, pydicom.multival.MultiValue):
        window_center = float(window_center[0])
    if isinstance(window_width, pydicom.multival.MultiValue):
        window_width = float(window_width[0])

    if window_center is None or window_width is None:
        # default: use the actual value range present in the image itself
        window_center = float((arr.max() + arr.min()) / 2)
        window_width = float(arr.max() - arr.min()) or 1.0
    else:
        window_center = float(window_center)
        window_width = float(window_width)

    lo = window_center - window_width / 2
    hi = window_center + window_width / 2

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo) * 255.0
    return arr.astype(np.uint8)


def convert_one_series(datasets: list, output_dir: str, max_slices: int = None):
    """Converts ONE series (a list of datasets sharing the same
    SeriesInstanceUID) into PNG slices + manifest.json in the given folder."""
    os.makedirs(output_dir, exist_ok=True)

    frames = collect_frames(datasets)
    frames = sort_frames_by_position(frames)

    total_available = len(frames)
    truncated = False
    if max_slices is not None and total_available > max_slices:
        # explicit, predictable limit -- take the first N slices AFTER
        # sorting by Z (i.e. from one physical edge of the volume),
        # without decimation. This is a deliberate prototype limitation,
        # see the project documentation.
        frames = frames[:max_slices]
        truncated = True

    first_ds = frames[0]["ds"]
    pixel_spacing = [float(x) for x in getattr(first_ds, "PixelSpacing", [1.0, 1.0])]
    slice_thickness = float(
        getattr(first_ds, "SpacingBetweenSlices", None)
        or getattr(first_ds, "SliceThickness", 1.0)
    )
    modality = str(getattr(first_ds, "Modality", "UNKNOWN"))
    series_description = str(getattr(first_ds, "SeriesDescription", "")).strip()

    manifest = {
        "modality": modality,
        "series_description": series_description,
        "pixel_spacing_mm": pixel_spacing,
        "slice_thickness_mm": slice_thickness,
        "slice_count": len(frames),
        "slice_count_available": total_available,
        "truncated": truncated,
        "slices": [],
    }

    z_positions = []

    for i, frame in enumerate(frames):
        ds = frame["ds"]
        pixel_array = frame["pixel_array"]
        normalized = apply_windowing(pixel_array, ds)

        img = Image.fromarray(normalized, mode="L")
        filename = f"slice_{i:04d}.png"
        img.save(os.path.join(output_dir, filename))

        z = frame["z"]
        z_positions.append(z)

        rows, cols = pixel_array.shape
        manifest["slices"].append({
            "index": i,
            "file": filename,
            "z_position_mm": round(z, 3),
            "rows": int(rows),
            "cols": int(cols),
        })

    if len(z_positions) > 1:
        diffs = np.diff(sorted(z_positions))
        manifest["z_step_mm_mean"] = round(float(np.mean(diffs)), 4)
        manifest["z_step_mm_std"] = round(float(np.std(diffs)), 4)

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  -> {len(frames)} PNG slices"
          + (f" (TRUNCATED from {total_available}, limit={max_slices})" if truncated else "")
          + f" + manifest.json -> {output_dir}")
    print(f"     modality: {modality}" + (f" ({series_description})" if series_description else ""))
    print(f"     pixel spacing (mm): {pixel_spacing}, slice thickness (mm): {slice_thickness}")
    print(f"     Z range: {min(z_positions):.2f} .. {max(z_positions):.2f} mm")
    if "z_step_mm_mean" in manifest:
        print(f"     mean Z step: {manifest['z_step_mm_mean']} mm "
              f"(std={manifest['z_step_mm_std']})")


def convert_one_series_quiet(datasets: list, output_dir: str, max_slices: int = None) -> dict:
    """Same as convert_one_series(), but without print() calls, and
    returns the manifest directly (manifest.json is still written to disk
    -- Lens needs it over HTTP -- but the calling backend code gets the
    content right away, without reading the file back)."""
    os.makedirs(output_dir, exist_ok=True)

    frames = collect_frames(datasets)
    frames = sort_frames_by_position(frames)

    total_available = len(frames)
    truncated = False
    if max_slices is not None and total_available > max_slices:
        frames = frames[:max_slices]
        truncated = True

    first_ds = frames[0]["ds"]
    pixel_spacing = [float(x) for x in getattr(first_ds, "PixelSpacing", [1.0, 1.0])]
    slice_thickness = float(
        getattr(first_ds, "SpacingBetweenSlices", None)
        or getattr(first_ds, "SliceThickness", 1.0)
    )
    modality = str(getattr(first_ds, "Modality", "UNKNOWN"))
    series_description = str(getattr(first_ds, "SeriesDescription", "")).strip()

    manifest = {
        "modality": modality,
        "series_description": series_description,
        "pixel_spacing_mm": pixel_spacing,
        "slice_thickness_mm": slice_thickness,
        "slice_count": len(frames),
        "slice_count_available": total_available,
        "truncated": truncated,
        "slices": [],
    }

    z_positions = []

    for i, frame in enumerate(frames):
        ds = frame["ds"]
        pixel_array = frame["pixel_array"]
        normalized = apply_windowing(pixel_array, ds)

        img = Image.fromarray(normalized, mode="L")
        filename = f"slice_{i:04d}.png"
        img.save(os.path.join(output_dir, filename))

        z = frame["z"]
        z_positions.append(z)

        rows, cols = pixel_array.shape
        manifest["slices"].append({
            "index": i,
            "file": filename,
            "z_position_mm": round(z, 3),
            "rows": int(rows),
            "cols": int(cols),
        })

    if len(z_positions) > 1:
        diffs = np.diff(sorted(z_positions))
        manifest["z_step_mm_mean"] = round(float(np.mean(diffs)), 4)
        manifest["z_step_mm_std"] = round(float(np.std(diffs)), 4)

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def convert_all_series(input_dir: str, output_dir: str, max_slices: int = None):
    """Finds all DICOM series in input_dir (including subfolders), groups
    them by SeriesInstanceUID, and lays each one out into its own subfolder
    inside output_dir.
    """
    print(f"Reading DICOM files from {input_dir} ...")
    datasets = load_series(input_dir)
    print(f"  found {len(datasets)} files")

    groups = group_by_series(datasets)
    print(f"  detected {len(groups)} series:\n")

    summaries = []
    for series_uid, series_datasets in groups.items():
        summaries.append(describe_series(series_uid, series_datasets))

    for idx, s in enumerate(summaries):
        desc = f" \"{s['series_description']}\"" if s["series_description"] else ""
        print(f"  [{idx}] {s['modality']}{desc} -- {s['file_count']} files, "
              f"{s['total_slices']} slices total, {s['rows']}x{s['cols']}px")

    print()
    os.makedirs(output_dir, exist_ok=True)

    for idx, (series_uid, series_datasets) in enumerate(groups.items()):
        s = summaries[idx]
        # folder name like 00_CT_AxialChest, no spaces or special characters
        safe_desc = "".join(
            c if c.isalnum() else "_" for c in (s["series_description"] or s["modality"])
        ).strip("_") or "series"
        series_output = os.path.join(output_dir, f"{idx:02d}_{safe_desc}")

        print(f"Series [{idx}] -> {series_output}")
        convert_one_series(series_datasets, series_output, max_slices=max_slices)
        print()

    print(f"Done. Result laid out into {len(groups)} subfolders in {output_dir}")


def convert_dicom_folder(input_dir: str, output_dir: str, max_slices: int = None) -> dict:
    """A side-effect-free version of convert_all_series() -- meant to be
    called from backend code (FastAPI), not from the console.

    Unlike convert_all_series():
      - prints nothing (no print() calls) -- instead it collects the same
        data into the returned structure, which the backend can log or put
        into a task status however it wants
      - converts ALL found series and returns a result list for each one,
        including the path to its subfolder (the caller decides whether
        to load one series or all of them, or to filter out localizers/
        movie series with its own heuristic on the backend side)
      - lets exceptions propagate as-is (RuntimeError from load_series if
        no files are found, etc.) -- the caller (BackgroundTasks in
        FastAPI) decides how to handle the error and what status to record

    Returns:
        a dict shaped like:
        {
            "series_count": int,
            "series": [
                {
                    "series_uid": str,
                    "output_dir": str,    # absolute path to the subfolder
                    "folder_name": str,    # subfolder name, e.g. "00_PD_SAG"
                    "manifest": {...}      # contents of the series' manifest.json
                },
                ...
            ]
        }
    """
    datasets = load_series(input_dir)
    groups = group_by_series(datasets)

    summaries = []
    for series_uid, series_datasets in groups.items():
        summaries.append(describe_series(series_uid, series_datasets))

    os.makedirs(output_dir, exist_ok=True)

    result_series = []
    for idx, (series_uid, series_datasets) in enumerate(groups.items()):
        s = summaries[idx]
        safe_desc = "".join(
            c if c.isalnum() else "_" for c in (s["series_description"] or s["modality"])
        ).strip("_") or "series"
        folder_name = f"{idx:02d}_{safe_desc}"
        series_output = os.path.join(output_dir, folder_name)

        manifest = convert_one_series_quiet(series_datasets, series_output, max_slices=max_slices)

        result_series.append({
            "series_uid": series_uid,
            "output_dir": series_output,
            "folder_name": folder_name,
            "manifest": manifest,
        })

    return {
        "series_count": len(result_series),
        "series": result_series,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts DICOM series into PNG slices + manifest.json for Spectacles"
    )
    parser.add_argument("input_dir", help="Folder with .dcm files (subfolders allowed)")
    parser.add_argument("output_dir", help="Folder for the result")
    parser.add_argument(
        "--max-slices",
        type=int,
        default=None,
        help="Maximum slices per series (prototype: performance limit on Spectacles). "
             "Without this flag, ALL slices are converted.",
    )
    args = parser.parse_args()

    convert_all_series(args.input_dir, args.output_dir, max_slices=args.max_slices)
