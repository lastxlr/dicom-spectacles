"""
main.py

FastAPI service for uploading DICOM files, converting them to PNG slices,
and uploading selected series to Supabase Storage under a short ID that
is later entered in the Lens on Spectacles.

Flow (see project_summary.md for full context):
  1. POST /upload      -- user uploads .dcm files. The service saves them
                           to a temp folder, generates an ID, starts
                           conversion IN THE BACKGROUND (BackgroundTasks),
                           and immediately returns {id, status: "processing"}.
  2. GET  /status/{id} -- the web page polls this endpoint every couple of
                           seconds. Once conversion is done, it returns the
                           list of found series (modality, description,
                           slice_count) -- the frontend builds checkboxes
                           from this.
  3. POST /select/{id} -- the user checks the series they want on the page,
                           the frontend sends a list of folder_name values.
                           The service uploads ONLY those series to
                           Supabase Storage under the same ID (in the
                           background) and marks the task "ready" once
                           the upload finishes.

Task state storage: a plain dict in process memory (TASKS). This is
sufficient for an MVP on a single Render worker -- if multiple workers or
restart-without-losing-state is ever needed, this would move to
Redis/a DB, but that's deliberately deferred as unnecessary complexity
for the MVP (see project_summary.md, BackgroundTasks section).

IMPORTANT: temp folders with DICOM/PNG files are NOT automatically
cleaned up after "ready" -- see the TODO at the end of this file about
periodic cleanup.
"""

import os
import shutil
import tempfile
import traceback
from typing import Literal

from dotenv import load_dotenv
load_dotenv()  # Reads .env from the current folder if present -- convenient
                # for local runs without manually setting environment
                # variables in PowerShell every time. On Render, variables
                # are set via Environment in the service settings; .env is
                # not needed there (and should not be committed to git).

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dicom_to_slices import convert_dicom_folder
from id_gen import generate_unique_id, generate_password
from supabase_upload import upload_series_to_supabase, create_signed_urls_for_folder

app = FastAPI(title="DICOM-to-Spectacles backend")

# Allow requests from any origin -- the upload web page may itself be
# served from a different domain/port than the backend (especially
# during development). For production this can be narrowed down to the
# actual domain of the upload page.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Max size per uploaded file (safety net -- Supabase free tier already
# caps uploads at 50MB per file, but it's better to reject early on receipt).
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB per file -- DICOM files
                                          # with many frames can be hefty

# Slice limit per series -- same cap that existed in the CLI version
# (--max-slices), kept here as a backend-side constant for consistency
# with the already-tested prototype.
MAX_SLICES_PER_SERIES = 40

# Max number of series the user can select in /select. Combined with
# MAX_SLICES_PER_SERIES, this caps the worst case at
# MAX_SLICES_PER_SERIES * MAX_SELECTED_SERIES = 240 planes in the Lens,
# which Spectacles can comfortably render.
MAX_SELECTED_SERIES = 6

TaskStatus = Literal["processing", "selecting", "uploading", "ready", "error"]


class TaskState(BaseModel):
    status: TaskStatus
    work_dir: str
    password: str
    series: list = []          # populated after conversion finishes
    selected_folders: list[str] = []  # populated after /select -- needed by
                                       # /access to know which folders'
                                       # files are signable
    error_message: str | None = None


# State of all tasks, kept in process memory. Key is the generated ID.
TASKS: dict[str, TaskState] = {}


def _validate_dcm_filename(filename: str) -> None:
    if not filename.lower().endswith(".dcm"):
        raise HTTPException(
            status_code=400,
            detail=f"File '{filename}' does not have a .dcm extension -- only DICOM files are accepted",
        )


def _run_conversion(task_id: str, input_dir: str, output_dir: str) -> None:
    """Runs in the background after /upload. Converts all series and moves
    the task to 'selecting' status, or to 'error' on failure."""
    try:
        result = convert_dicom_folder(input_dir, output_dir, max_slices=MAX_SLICES_PER_SERIES)

        # Build a compact summary per series -- this is what the frontend
        # sees to build checkboxes; the full manifest (with the slice list)
        # is unnecessary at this stage.
        series_summary = []
        for s in result["series"]:
            m = s["manifest"]
            series_summary.append({
                "folder_name": s["folder_name"],
                "modality": m["modality"],
                "series_description": m["series_description"],
                "slice_count": m["slice_count"],
                "truncated": m["truncated"],
            })

        TASKS[task_id].series = series_summary
        TASKS[task_id].status = "selecting"
    except Exception as e:
        TASKS[task_id].status = "error"
        TASKS[task_id].error_message = f"Conversion error: {e}"
        print(f"[{task_id}] CONVERSION ERROR:\n{traceback.format_exc()}")


def _run_upload_to_supabase(task_id: str, selected_folders: list[str]) -> None:
    """Runs in the background after /select. Uploads the selected series to
    Supabase Storage under task_id and moves the task to 'ready' or 'error'."""
    task = TASKS[task_id]
    try:
        root_manifest_series = []
        for folder_name in selected_folders:
            series_dir = os.path.join(task.work_dir, "converted", folder_name)
            if not os.path.isdir(series_dir):
                raise RuntimeError(f"Series folder '{folder_name}' not found -- "
                                    f"invalid folder_name from client")

            upload_series_to_supabase(
                local_series_dir=series_dir,
                remote_prefix=f"{task_id}/{folder_name}",
            )

            # Read this series' summary to include it in the root manifest
            # (uploaded as a separate step right after this loop).
            series_entry = next(s for s in task.series if s["folder_name"] == folder_name)
            root_manifest_series.append(series_entry)

        # Root manifest.json -- this is what Lens fetches first by ID to
        # find out which series (subfolders) are available under this ID.
        # Right now Lens in the MVP always picks series[0] (see
        # project_summary.md), but the structure is already set up for a
        # future selection step on the Lens side (variant B).
        import json
        root_manifest = {"id": task_id, "series": root_manifest_series}
        root_manifest_path = os.path.join(task.work_dir, "manifest.json")
        with open(root_manifest_path, "w") as f:
            json.dump(root_manifest, f, indent=2)

        upload_series_to_supabase(
            local_series_dir=task.work_dir,
            remote_prefix=task_id,
            only_files=["manifest.json"],
        )

        task.status = "ready"
    except Exception as e:
        task.status = "error"
        task.error_message = f"Supabase upload error: {e}"
        print(f"[{task_id}] UPLOAD ERROR:\n{traceback.format_exc()}")


@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
    """Accepts .dcm files, saves them to a temp folder, starts conversion
    in the background, and immediately returns an ID for polling."""
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided")

    for f in files:
        _validate_dcm_filename(f.filename)

    task_id = generate_unique_id(TASKS.keys())

    work_dir = tempfile.mkdtemp(prefix=f"dicom_{task_id}_")
    input_dir = os.path.join(work_dir, "input")
    output_dir = os.path.join(work_dir, "converted")
    os.makedirs(input_dir, exist_ok=True)

    total_size = 0
    for f in files:
        contents = await f.read()
        total_size += len(contents)
        if len(contents) > MAX_FILE_SIZE_BYTES:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(
                status_code=413,
                detail=f"File '{f.filename}' exceeds the {MAX_FILE_SIZE_BYTES // (1024*1024)}MB limit",
            )
        dest_path = os.path.join(input_dir, f.filename)
        with open(dest_path, "wb") as out:
            out.write(contents)

    TASKS[task_id] = TaskState(
        status="processing",
        work_dir=work_dir,
        password=generate_password(),
    )

    background_tasks.add_task(_run_conversion, task_id, input_dir, output_dir)

    return {"id": task_id, "password": TASKS[task_id].password, "status": "processing"}


@app.get("/status/{task_id}")
async def status(task_id: str):
    """Returns the current task status. Once conversion is done, the
    response includes the list of series for the frontend to show."""
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="ID not found")

    response = {"status": task.status}
    if task.status in ("selecting", "uploading", "ready"):
        response["series"] = task.series
    if task.status == "error":
        response["error"] = task.error_message
    return response


class SelectRequest(BaseModel):
    folder_names: list[str]


@app.post("/select/{task_id}")
async def select(task_id: str, body: SelectRequest, background_tasks: BackgroundTasks):
    """Accepts the list of selected series (folder_name values from
    /status) and starts the Supabase upload in the background."""
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="ID not found")
    if task.status != "selecting":
        raise HTTPException(
            status_code=409,
            detail=f"Task is in status '{task.status}', series selection is not possible right now",
        )
    if not body.folder_names:
        raise HTTPException(status_code=400, detail="At least one series must be selected")
    if len(body.folder_names) > MAX_SELECTED_SERIES:
        raise HTTPException(
            status_code=400,
            detail=f"At most {MAX_SELECTED_SERIES} series can be selected at once "
                    f"(got {len(body.folder_names)})",
        )

    valid_names = {s["folder_name"] for s in task.series}
    unknown = [name for name in body.folder_names if name not in valid_names]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown series: {unknown}")

    task.status = "uploading"
    task.selected_folders = body.folder_names
    background_tasks.add_task(_run_upload_to_supabase, task_id, body.folder_names)

    return {"id": task_id, "status": "uploading"}


@app.get("/health")
async def health():
    """Simple healthcheck -- useful for Render to verify the service is
    alive, without any side effects."""
    return {"status": "ok"}


# Same default as create_signed_url()'s expires_in_seconds -- kept as a
# constant here too so both sides of the contract are easy to find.
SIGNED_URL_EXPIRY_SECONDS = 7200  # 2 hours -- long enough for a single
                                   # Spectacles viewing session


@app.get("/access/{task_id}")
async def access(task_id: str, password: str):
    """Verifies the ID + password pair and, if correct, returns signed
    Storage URLs for the root manifest.json and for every series'
    manifest.json + PNG slices that were uploaded for this task.

    This is the endpoint Lens calls on Spectacles -- it does NOT build
    Storage URLs itself from a known pattern (the bucket is private, so a
    guessed/predicted URL would not work without a valid token anyway).

    Response shape:
        {
            "root_manifest_url": "https://.../manifest.json?token=...",
            "series": [
                {
                    "folder_name": "00_PD_SAG",
                    "manifest_url": "https://.../00_PD_SAG/manifest.json?token=...",
                    "slice_urls": ["https://.../00_PD_SAG/slice_0000.png?token=...", ...]
                },
                ...
            ]
        }
    """
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="ID not found")

    # Constant-time-ish comparison isn't critical here -- this is a
    # convenience PIN, not a cryptographic secret (see id_gen.py) -- a
    # plain == is fine and keeps the code simple.
    if password != task.password:
        raise HTTPException(status_code=403, detail="Incorrect password")

    if task.status != "ready":
        raise HTTPException(
            status_code=409,
            detail=f"Task is in status '{task.status}', files are not uploaded yet",
        )

    root_url = create_signed_urls_for_folder(
        local_filenames=["manifest.json"],
        remote_prefix=task_id,
        expires_in_seconds=SIGNED_URL_EXPIRY_SECONDS,
    )["manifest.json"]

    series_response = []
    for folder_name in task.selected_folders:
        series_entry = next(s for s in task.series if s["folder_name"] == folder_name)
        slice_count = series_entry["slice_count"]
        slice_filenames = [f"slice_{i:04d}.png" for i in range(slice_count)]

        signed = create_signed_urls_for_folder(
            local_filenames=["manifest.json"] + slice_filenames,
            remote_prefix=f"{task_id}/{folder_name}",
            expires_in_seconds=SIGNED_URL_EXPIRY_SECONDS,
        )

        series_response.append({
            "folder_name": folder_name,
            "manifest_url": signed["manifest.json"],
            "slice_urls": [signed[fname] for fname in slice_filenames],
        })

    return {
        "root_manifest_url": root_url,
        "series": series_response,
    }


# TODO (not critical for the MVP, but important not to forget before real
# use at the competition): temp folders (work_dir) in TASKS are never
# deleted. With sustained use, /tmp on Render will gradually fill up.
# Future fix: a periodic job (e.g. APScheduler, or simply checking folder
# age on every new /upload) that cleans up work_dir for tasks older than
# N hours. For a single run at the competition this isn't a real problem,
# but if the service stays up for more than a couple of days, this should
# be added.
