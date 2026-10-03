"""Frozen Protocol v1 reference permutation (no parameters or learned state)."""
import hmac
import struct
import numpy as np
import torch
from .crypto import COLOR_INFO, PATIENT_INFO, derive_key, require_bytes
from .config import finite_nonnegative

COLOR, PATIENT = 1, 2
LAYERS = {COLOR: tuple(range(5, 11)), PATIENT: (3, 4)}
COORDINATES = {branch: tuple((u, q-u) for q in qs for u in range(8) if 0 <= q-u < 8)
               for branch, qs in LAYERS.items()}
CHANNELS = {branch: tuple(u * 8 + v for u, v in coords) for branch, coords in COORDINATES.items()}


def permutation_key(master, salt, branch):
    if branch not in LAYERS:
        raise ValueError("Unknown branch")
    return derive_key(master, salt, COLOR_INFO if branch == COLOR else PATIENT_INFO)


def prf_context(image_id, branch, row, col, q, r):
    require_bytes(image_id, 16, "ImageID")
    if branch not in LAYERS or q not in LAYERS[branch] or not 0 <= row < 32 or not 0 <= col < 32 or not 1 <= r <= 15:
        raise ValueError("Invalid PRF context")
    return b"DPWPERM1" + image_id + struct.pack(">BHHBB", branch, row, col, q, r)


class PRFStream:
    def __init__(self, key, context):
        require_bytes(key, 32, "permutation key")
        require_bytes(context, 31, "PRF context")
        self.key, self.context = key, context
        self.counter, self.buffer, self.position = 0, b"", 0

    def byte(self):
        if self.position == len(self.buffer):
            if self.counter >= 2**32:
                raise ValueError("PRF counter exhausted")
            self.buffer = hmac.digest(self.key, self.context + self.counter.to_bytes(4, "big"), "sha256")
            self.counter += 1
            self.position = 0
        result = self.buffer[self.position]
        self.position += 1  # Every byte is consumed, including rejected bytes.
        return result


def fisher_yates(size, stream):
    pi = list(range(size))
    for i in range(size - 1, 0, -1):
        length = i + 1
        limit = (256 // length) * length  # Python int represents 256.
        value = stream.byte()
        while value >= limit:
            value = stream.byte()
        j = value % length
        pi[i], pi[j] = pi[j], pi[i]
    return tuple(pi)


def candidate(key, image_id, branch, row, col, r):
    """One r determines ALL q layers. Duplicates are kept, never resampled."""
    if branch not in LAYERS or type(r) is not int or not 0 <= r <= 15:
        raise ValueError("Invalid branch or selector")
    if r == 0:
        return tuple(range(len(CHANNELS[branch])))
    result = []
    for q in LAYERS[branch]:
        size = sum(0 <= q-u < 8 for u in range(8))
        stream = PRFStream(key, prf_context(image_id, branch, row, col, q, r))
        offset = len(result)
        result.extend(offset + index for index in fisher_yates(size, stream))
    return tuple(result)


def distortion(values, pi):
    """Sequential binary64 subtract, multiply, add; no vector reduction/FMA."""
    total = 0.0
    for i in range(len(pi)):
        difference = float(values[pi[i]]) - float(values[i])
        square = difference * difference
        total = total + square
    return total


def select_candidate(values, key, image_id, branch, row, col, min_moved, beta):
    maximum = len(CHANNELS[branch])
    if type(min_moved) is not int or not 1 <= min_moved <= maximum:
        raise ValueError("Invalid min_moved")
    finite_nonnegative(beta, "beta")
    beta = float(beta)  # Compare the same binary64 value that is encoded in H0.
    values = np.asarray(values)
    if values.dtype != np.float32 or values.shape != (maximum,) or not np.isfinite(values).all():
        raise ValueError("Candidate selection requires finite FP32 branch coefficients")
    best = None
    for r in range(1, 16):
        pi = candidate(key, image_id, branch, row, col, r)
        moved = sum(i != source for i, source in enumerate(pi))
        if moved >= min_moved:
            cost = distortion(values, pi)
            if best is None or cost < best[0]:  # Ascending r resolves ties.
                best = (cost, r, pi)
    if best is None or best[0] > beta:
        return 0, tuple(range(maximum)), 0.0
    return best[1], best[2], best[0]


def pack_selectors(color, patient):
    if len(color) != 1024 or len(patient) != 1024:
        raise ValueError("Each branch requires 1024 selectors")
    if any(type(x) is not int or not 0 <= x <= 15 for x in (*color, *patient)):
        raise ValueError("Selectors must be integers 0..15")
    return bytes((c << 4) | m for c, m in zip(color, patient))


def unpack_selectors(raw):
    require_bytes(raw, 1024, "selector")
    return tuple(x >> 4 for x in raw), tuple(x & 15 for x in raw)


def permute_coefficients(coefficients, color_key, patient_key, header):
    """Single-image send. Both branches read original C, write independent slots."""
    if coefficients.shape != (1, 64, 32, 32) or coefficients.dtype != torch.float32:
        raise ValueError("Expected FP32 [1,64,32,32] coefficients")
    source = coefficients.detach().cpu().numpy()[0]
    if not np.isfinite(source).all():
        raise ValueError("Nonfinite DCT coefficients")
    result = source.copy()
    selected = {COLOR: [], PATIENT: []}
    for branch, key, minimum, beta in (
        (COLOR, color_key, header.min_moved_c, header.beta_c),
        (PATIENT, patient_key, header.min_moved_m, header.beta_m),
    ):
        indices = list(CHANNELS[branch])
        for row in range(32):
            for col in range(32):
                values = source[indices, row, col]
                r, pi, _ = select_candidate(values, key, header.image_id, branch, row, col, minimum, beta)
                result[indices, row, col] = values[list(pi)]
                selected[branch].append(r)
    tensor = torch.from_numpy(result).unsqueeze(0).to(coefficients.device)
    return tensor, pack_selectors(selected[COLOR], selected[PATIENT])


def inverse_branch(coefficients, key, image_id, branch, selectors):
    """Recover branch-only features [1,39/9,32,32], without changing other bands."""
    size = len(CHANNELS[branch])
    if coefficients.shape != (1, size, 32, 32) or coefficients.dtype != torch.float32:
        raise ValueError("Unexpected branch coefficients")
    if not bool(torch.isfinite(coefficients).all()):
        raise ValueError("Nonfinite received coefficients")
    color, patient = unpack_selectors(selectors)
    codes = color if branch == COLOR else patient
    source = coefficients.detach().cpu().numpy()[0]
    result = np.empty_like(source)
    for b, r in enumerate(codes):
        row, col = divmod(b, 32)
        pi = candidate(key, image_id, branch, row, col, r)
        result[list(pi), row, col] = source[:, row, col]  # Independent buffer.
    return torch.from_numpy(result).unsqueeze(0).to(coefficients.device)
