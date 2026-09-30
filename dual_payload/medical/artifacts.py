"""Checksummed experiment records and explicit checkpoint provenance."""
import hashlib
import io
import json
from pathlib import Path

import torch

from .profile import canonical_json, read_json


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def code_digest():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def save_record(path, document):
    document = dict(document)
    document.pop('sha256', None)
    document['sha256'] = hashlib.sha256(canonical_json(document)).hexdigest()
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return document


def load_record(path):
    document = read_json(path)
    checksum = document.pop('sha256')
    if hashlib.sha256(canonical_json(document)).hexdigest() != checksum:
        raise ValueError(f'Experiment record checksum mismatch: {path}')
    document['sha256'] = checksum
    return document


def checked_checkpoint(path, expected_sha256):
    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError('Explicit checkpoint path and verified SHA-256 are required')
    raw = Path(path).read_bytes()
    if not isinstance(expected_sha256, str) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Checkpoint SHA-256 mismatch; provide its verified digest')
    result = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    if not isinstance(result, dict):
        raise ValueError('Expected checkpoint dictionary')
    return result


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)
