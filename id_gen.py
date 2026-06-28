"""
id_gen.py

Generates a short random ID and a short numeric password for a DICOM
upload/series bundle.

Requirements (per discussion with the user):
  - ID: short, easy to type by hand on Spectacles (hand-tracking / virtual
    keyboard, not a physical keyboard) -- hence 6 characters, not a UUID
  - ID: no visually similar characters, so the user doesn't mix them up
    while typing: 0/O and 1/I/L are removed to avoid confusion on the
    small glasses display
  - ID: uppercase letters and digits only -- no need to switch case on a
    virtual keyboard
  - password: a separate short numeric PIN, used together with the ID to
    request signed Supabase Storage URLs from the backend (the bucket is
    private -- see supabase_upload.py). Digits only, since a numeric
    keypad is even faster to use on Spectacles than the full alphabet.
"""

import secrets

# Alphabet without 0, O, 1, I, L -- visually ambiguous characters.
_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_ID_LENGTH = 6

_PASSWORD_LENGTH = 4


def generate_id() -> str:
    """Generates a random ID like 'A3K9F2'."""
    return "".join(secrets.choice(_ALPHABET) for _ in range(_ID_LENGTH))


def generate_unique_id(existing_ids) -> str:
    """Generates an ID guaranteed not to collide with existing_ids (a
    set/collection of already-used IDs). With 32^6 ~ 1 billion possible
    combinations, collisions are extremely rare, but the check is cheap
    enough to always perform rather than rely on statistics alone."""
    for _ in range(100):
        candidate = generate_id()
        if candidate not in existing_ids:
            return candidate
    raise RuntimeError("Could not generate a unique ID in 100 attempts -- "
                        "the ID space is nearly exhausted, _ID_LENGTH needs to be increased")


def generate_password() -> str:
    """Generates a random 4-digit numeric password, e.g. '0426'.

    This is paired with the ID to gate access to signed Storage URLs.
    It is NOT meant to resist a sustained brute-force attack (10,000
    combinations is small) -- its purpose is to prevent casual/accidental
    access by someone who only has the ID (e.g. saw it on a screen or
    intercepted Lens network traffic), not to be cryptographically secure.
    """
    return "".join(secrets.choice("0123456789") for _ in range(_PASSWORD_LENGTH))

