"""
supabase_upload.py

Uploads local files (PNG slices + manifest.json) to Supabase Storage
using a service-role (secret) key.

IMPORTANT about security (see project_summary.md): the service-role key
grants full write access to Storage, bypassing Row Level Security. It must
live ONLY in the backend service's environment variables (on Render --
via Environment Variables in the service settings) and must never end up
in client-side code (the upload web page or the Lens).

Dependency: the `supabase` package (official Python SDK).
Install: pip install supabase

Environment variables that must be set on Render:
  SUPABASE_URL              -- project URL, e.g. https://xxxxx.supabase.co
  SUPABASE_SERVICE_ROLE_KEY -- service-role / secret key (Settings -> API
                                in the dashboard)
  SUPABASE_BUCKET           -- bucket name in Storage (must be created
                                beforehand in the Supabase dashboard, with
                                public read access)
"""

import os
import mimetypes

from supabase import create_client, Client

_SUPABASE_URL = os.environ.get("SUPABASE_URL")
_SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
_SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "dicom-slices")

_client: Client | None = None


def _get_client() -> Client:
    """Lazy client initialization -- so the module can be imported (e.g. in
    tests) even without the environment variables set, and the error only
    occurs on an actual upload attempt."""
    global _client
    if _client is None:
        if not _SUPABASE_URL or not _SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in "
                "the backend service's environment variables (Render -> Environment)"
            )
        _client = create_client(_SUPABASE_URL, _SUPABASE_SERVICE_ROLE_KEY)
    return _client


def upload_series_to_supabase(
    local_series_dir: str,
    remote_prefix: str,
    only_files: list[str] | None = None,
) -> None:
    """Uploads all files from local_series_dir to Supabase Storage under
    remote_prefix/<filename>.

    Args:
        local_series_dir: local folder with files (PNG slices + manifest.json
            for one series, or the root work_dir when uploading the root
            manifest.json)
        remote_prefix: path inside the bucket under which the files will
            end up, e.g. "A3K9F2/00_PD_SAG" -- then files will be reachable
            at {SUPABASE_URL}/storage/v1/object/public/{bucket}/A3K9F2/00_PD_SAG/slice_0000.png
        only_files: if given -- only files with these names are uploaded
            (instead of the entire folder contents). Used to upload the
            root manifest.json separately from the series subfolders.
    """
    client = _get_client()

    filenames = only_files if only_files is not None else os.listdir(local_series_dir)

    for filename in filenames:
        local_path = os.path.join(local_series_dir, filename)
        if not os.path.isfile(local_path):
            continue

        remote_path = f"{remote_prefix}/{filename}"
        content_type, _ = mimetypes.guess_type(filename)
        content_type = content_type or "application/octet-stream"

        with open(local_path, "rb") as f:
            data = f.read()

        # upsert=true in case of a repeated upload under the same ID (e.g.
        # if /select is somehow called twice). cacheControl is left at a
        # moderate default -- project_summary.md notes a CDN caching
        # quirk when re-uploading files under the same name, but for the
        # MVP (a single upload per ID, no re-uploads) this isn't critical.
        # See project_summary.md for the filename-versioning note if
        # re-uploads are ever needed.
        client.storage.from_(_SUPABASE_BUCKET).upload(
            path=remote_path,
            file=data,
            file_options={"content-type": content_type, "upsert": "true"},
        )


def get_public_url(remote_path: str) -> str:
    """Returns the public HTTPS URL of a file. Only works if the bucket is
    public -- kept here for reference/diagnostics, but the service now
    uses a PRIVATE bucket (see create_signed_url below) gated by an
    ID + password pair."""
    client = _get_client()
    return client.storage.from_(_SUPABASE_BUCKET).get_public_url(remote_path)


def create_signed_url(remote_path: str, expires_in_seconds: int = 7200) -> str:
    """Creates a temporary signed URL for a single file in a PRIVATE
    bucket. The URL embeds a token and works without any further
    authentication until it expires -- this is what Lens uses to download
    manifest.json and PNG slices after the backend has verified the
    ID + password pair (see /access in main.py).

    Args:
        remote_path: path inside the bucket, e.g. "A3K9F2/00_PD_SAG/slice_0000.png"
        expires_in_seconds: how long the URL stays valid. Default 2 hours --
            long enough for a single Spectacles viewing session, short
            enough that a leaked URL doesn't stay usable indefinitely.

    Returns:
        the full signed HTTPS URL as a string.
    """
    client = _get_client()
    response = client.storage.from_(_SUPABASE_BUCKET).create_signed_url(
        path=remote_path,
        expires_in=expires_in_seconds,
    )
    # The Python SDK returns a dict with a "signedURL" key (relative path,
    # without the project domain) -- build the absolute URL ourselves so
    # the caller doesn't need to know this detail.
    signed_path = response["signedURL"]
    if signed_path.startswith("http"):
        return signed_path
    return f"{_SUPABASE_URL}/storage/v1{signed_path}" if not signed_path.startswith("/storage") \
        else f"{_SUPABASE_URL}{signed_path}"


def create_signed_urls_for_folder(local_filenames: list[str], remote_prefix: str,
                                    expires_in_seconds: int = 7200) -> dict[str, str]:
    """Creates signed URLs for a known list of filenames under remote_prefix.

    Storage doesn't support signing an entire folder with one call, so
    this signs each file individually. Returns a dict mapping filename ->
    signed URL, so the caller (the /access endpoint) can hand Lens a
    ready-to-use lookup table instead of a path template it would have to
    guess at (which wouldn't work anyway, since signed URLs include a
    one-time token that can't be predicted).
    """
    return {
        filename: create_signed_url(f"{remote_prefix}/{filename}", expires_in_seconds)
        for filename in local_filenames
    }
