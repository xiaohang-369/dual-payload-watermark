"""Validated public profiles and immutable local registration (no secrets)."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .protocol import BLOCKS, COLOR_INDICES, PATIENT_INDICES, Branch


ARCHITECTURE = "medical-v1"
FIXED = {
    "schema": 1,
    "image": {"width": 256, "height": 256, "format": "gray8",
              "rgb": "gamma-srgb-bt601-full-zero-chroma",
              "integer_output": "clip-0-1-times-255-round-even"},
    "dct": {"basis": "orthonormal-dct-ii-8", "color_indices": list(COLOR_INDICES),
            "patient_indices": list(PATIENT_INDICES), "order": "block-row-column-frequency"},
    "quantizer": {"bits": 4, "min": -8, "max": 7, "round": "ties-even"},
    "ldpc": {"implementation": "sionna", "version": "2.1.0", "k": 1024, "n": 1536,
             "bg": "bg2", "rv": 0, "qam_interleaver": False, "harq": False,
             "cn_update": "boxplus-phi", "schedule": "flooding", "iterations": 20,
             "llr_max": 20.0, "positive_logits": "bit1"},
    "layout": {"order": "channel-row-column", "color_channels": 236,
               "patient_channels": 2, "color_blocks": 157, "patient_blocks": 1,
               "color_frame_bytes": 20052, "patient_frame_bytes": 76,
               "data_padding": "zero", "layout_padding_bits": 512,
               "interleave": "output[i]=codeword[permutation[i]]"},
    "preprocess": {"orientation": "apply-exif-before-resize", "resize": "pillow-bicubic",
                   "geometry": "fit-long-side-256-round-even-center-pad",
                   "padding": "edge", "padding_embeddable": True,
                   "input": "RGB8-or-opaque-RGBA8",
                   "untagged_color": "assume-sRGB",
                   "icc": "embedded-RGB-to-sRGB-relative-colorimetric-flags-0-before-resize"},
    "architecture": ARCHITECTURE,
}


def canonical_json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_pairs)


def template() -> dict:
    result = json.loads(canonical_json(FIXED))
    result.update(profile_id=None, quantization_id=None, purpose="experiment",
                  quantization_steps=None,
                  rms_limits={"color_residual": None, "color_ciphertext": None,
                              "patient_ciphertext": None},
                  weights={name: None for name in ("ec", "ew", "dc", "dw")},
                  interleaver_seeds={"color": 20260930, "patient": 20260931})
    return result


def _uint16(value, name):
    if type(value) is not int or not 1 <= value <= 65535:
        raise ValueError(f"{name} must be an integer in [1,65535]")


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_config(config: dict, *, allow_test: bool = False) -> None:
    if not isinstance(config, dict) or set(config) != set(template()):
        raise ValueError("Missing or unknown medical profile fields")
    for name, value in FIXED.items():
        if canonical_json({name: config[name]}) != canonical_json({name: value}):
            raise ValueError(f"Unsupported medical-v1 setting: {name}")
    for name in ("profile_id", "quantization_id"):
        _uint16(config[name], name)
    if config["purpose"] not in ("experiment", "test"):
        raise ValueError("Unknown profile purpose")
    if config["purpose"] == "test" and not allow_test:
        raise ValueError("Test profile requires explicit --allow-test-profile")
    steps = config["quantization_steps"]
    if not isinstance(steps, list) or len(steps) != 39 or any(
        type(v) not in (int, float) or not math.isfinite(v) or not 1e-30 <= v <= 1e30 for v in steps
    ):
        raise ValueError("39 finite positive calibrated quantization_steps are required")
    limits = config["rms_limits"]
    if not isinstance(limits, dict) or set(limits) != {
        "color_residual", "color_ciphertext", "patient_ciphertext"
    } or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1
             for v in limits.values()):
        raise ValueError("All three finite RMS limits must be explicitly set in (0,1]")
    weights = config["weights"]
    if not isinstance(weights, dict) or set(weights) != {"ec", "ew", "dc", "dw"} or not all(
        _sha(v) for v in weights.values()
    ):
        raise ValueError("SHA-256 of each ec/ew/dc/dw weight file is required")
    seeds = config["interleaver_seeds"]
    if not isinstance(seeds, dict) or set(seeds) != {"color", "patient"} or any(
        type(v) is not int or not 0 <= v < 2**64 for v in seeds.values()
    ):
        raise ValueError("Two uint64 interleaver seeds are required")


@dataclass(frozen=True)
class Profile:
    """Keep canonical bytes internally, so callers cannot mutate the configuration."""

    canonical: bytes
    permutations: tuple[np.ndarray, np.ndarray]
    inverses: tuple[np.ndarray, np.ndarray]

    @property
    def document(self):
        return json.loads(self.canonical)

    @property
    def config(self):
        return self.document["config"]

    @property
    def profile_id(self):
        return self.config["profile_id"]

    @property
    def digest(self):
        return hashlib.sha256(self.canonical).digest()

    @property
    def steps(self):
        return tuple(self.config["quantization_steps"])

    def model_id(self, names):
        return hashlib.sha256(b"".join(bytes.fromhex(self.config["weights"][n]) for n in names)).digest()

    def permutation(self, branch: Branch, inverse=False):
        return (self.inverses if inverse else self.permutations)[int(branch) - 1]


def register_profile(config: dict, registry: str | Path, *, allow_test=False) -> Path:
    """Generate arrays once and freeze their bytes and checksums with the config."""
    validate_config(config, allow_test=allow_test)
    directory = Path(registry) / str(config["profile_id"])
    if directory.exists():
        existing = load_profile(directory, allow_test=allow_test)
        if canonical_json(existing.config) != canonical_json(config):
            raise ValueError("Profile ID is already registered to a different configuration")
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for branch, name in ((Branch.COLOR, "color"), (Branch.PATIENT, "patient")):
        size = BLOCKS[branch] * 1536
        rng = np.random.Generator(np.random.PCG64(config["interleaver_seeds"][name]))
        permutation = rng.permutation(size)
        inverse = np.argsort(permutation)
        # Each LDPC block must occupy multiple spatial locations after layout.
        if any(np.unique(inverse[i:i + 1536] % 1024).size < 2 for i in range(0, size, 1536)):
            raise ValueError("Interleaver does not disperse code blocks")
        for suffix, array in (("permutation", permutation), ("inverse", inverse)):
            filename = f"{name}-{suffix}.u32be"
            data = array.astype(">u4").tobytes()
            (directory / filename).write_bytes(data)
            manifest[filename] = hashlib.sha256(data).hexdigest()
    data = canonical_json({"config": config, "arrays": manifest})
    (directory / "profile.json").write_bytes(data)
    (directory / "profile.sha256").write_text(hashlib.sha256(data).hexdigest() + "\n")
    return directory


def load_profile(directory: str | Path, *, allow_test=False) -> Profile:
    directory = Path(directory)
    document = read_json(directory / "profile.json")
    if not isinstance(document, dict) or set(document) != {"config", "arrays"}:
        raise ValueError("Invalid registered profile")
    data = canonical_json(document)
    if hashlib.sha256(data).hexdigest() != (directory / "profile.sha256").read_text().strip():
        raise ValueError("Registered profile checksum changed")
    validate_config(document["config"], allow_test=allow_test)
    expected_names = {f"{name}-{kind}.u32be" for name in ("color", "patient")
                      for kind in ("permutation", "inverse")}
    if set(document["arrays"]) != expected_names:
        raise ValueError("Invalid interleaver manifest")
    permutations, inverses = [], []
    for branch, name in ((Branch.COLOR, "color"), (Branch.PATIENT, "patient")):
        arrays = []
        size = BLOCKS[branch] * 1536
        for suffix in ("permutation", "inverse"):
            filename = f"{name}-{suffix}.u32be"
            raw = (directory / filename).read_bytes()
            if len(raw) != size * 4 or hashlib.sha256(raw).hexdigest() != document["arrays"][filename]:
                raise ValueError("Interleaver length or checksum mismatch")
            array = np.frombuffer(raw, dtype=">u4").astype(np.int64)
            if not np.array_equal(np.sort(array), np.arange(size)):
                raise ValueError("Interleaver is not a permutation")
            array.setflags(write=False)
            arrays.append(array)
        if not np.array_equal(arrays[0][arrays[1]], np.arange(size)):
            raise ValueError("Inverse interleaver mismatch")
        permutations.append(arrays[0])
        inverses.append(arrays[1])
    return Profile(data, tuple(permutations), tuple(inverses))
