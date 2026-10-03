"""Algorithms 1–3, bound to the exact bytes of an existing model checkpoint."""
from contextlib import contextmanager
import secrets
import torch
from .checkpoints import model_from_file
from .config import validate_protocol
from .crypto import (require_bytes, encrypt_patient, decrypt_patient,
                     payload_to_bits, logits_to_payload)
from .keyed_permutation import (COLOR, PATIENT, CHANNELS, permutation_key,
                                permute_coefficients, inverse_branch)
from .package import Header, encode_package, verify_package


@contextmanager
def fp32_inference(device):
    """Disable mixed precision and TF32, restoring caller settings afterwards."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


class ProtocolV1:
    def __init__(self, checkpoint_path, *, device="cpu"):
        if torch.device(device).type not in ("cpu", "cuda"):
            raise ValueError("Protocol v1 implementation supports CPU/CUDA FP32")
        self.model, self.config, self.model_id = model_from_file(checkpoint_path, device)
        self.model.eval().requires_grad_(False)
        self.device = next(self.model.parameters()).device
        # Detect later accidental in-place changes to this exact checkpoint instance.
        self._versions = [(value, value._version) for value in (*self.model.parameters(), *self.model.buffers())]

    def _check_model(self):
        if any(module.training for module in self.model.modules()) or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("Protocol model must remain frozen in eval mode")
        current = (*self.model.parameters(), *self.model.buffers())
        if len(current) != len(self._versions) or any(
            value is not original or value._version != version
            for value, (original, version) in zip(current, self._versions)
        ):
            raise ValueError("Protocol model changed after checkpoint binding")

    def publish(self, rgb, patient_token, kc, km, signing_key, parameters):
        package, _ = self._publish_for_evaluation(rgb, patient_token, kc, km, signing_key, parameters)
        return package

    def _publish_for_evaluation(self, rgb, patient_token, kc, km, signing_key, parameters):
        """Return sent bits privately for BER; never encrypt again with the same nonce."""
        self._check_model()
        validate_protocol(parameters)
        require_bytes(patient_token, 16, "PatientToken")
        require_bytes(kc, 32, "KC")
        require_bytes(km, 32, "KM")
        if rgb.shape != (1, 3, 256, 256) or rgb.dtype != torch.float32 or not bool(torch.isfinite(rgb).all()) or not bool(((rgb >= 0) & (rgb <= 1)).all()):
            raise ValueError("Protocol requires one finite FP32 sRGB image [1,3,256,256] in [0,1]")
        header = Header(image_id=secrets.token_bytes(16), model_id=self.model_id,
                        beta_c=parameters["beta_c"], beta_m=parameters["beta_m"],
                        min_moved_c=parameters["min_moved_c"], min_moved_m=parameters["min_moved_m"],
                        nc=secrets.token_bytes(16), nm=secrets.token_bytes(16), ngcm=secrets.token_bytes(12))
        payload = encrypt_patient(patient_token, km, header.nm, header.ngcm, header.encode())
        ckey = permutation_key(kc, header.nc, COLOR)
        mkey = permutation_key(km, header.nm, PATIENT)
        with fp32_inference(self.device):
            message = payload_to_bits(payload).unsqueeze(0).to(self.device)
            output = self.model.encode(rgb.to(self.device), message)
            dct = self.model.color_encoder.dct
            coefficients = dct(output["x_float"])
            permuted, selectors = permute_coefficients(coefficients, ckey, mkey, header)
            gray = dct.inverse(permuted)
            return encode_package(header, selectors, gray, signing_key), message

    def _verify(self, data, trusted_public_key):
        # Authentication precedes header semantics, key use, DCT or decoder calls.
        verified = verify_package(data, trusted_public_key, expected_model_id=self.model_id)
        self._check_model()
        return verified

    def recover_color(self, data, kc, trusted_public_key):
        verified = self._verify(data, trusted_public_key)
        header = verified.header
        key = permutation_key(kc, header.nc, COLOR)
        with fp32_inference(self.device):
            decoder = self.model.color_decoder
            coefficients = decoder.dct(verified.gray.to(self.device))
            if not bool(torch.isfinite(coefficients).all()):
                raise ValueError("Nonfinite received DCT coefficients")
            indices = list(CHANNELS[COLOR])
            restored = inverse_branch(coefficients[:, indices], key, header.image_id, COLOR, verified.raw.selectors)
            combined = coefficients.clone()
            combined[:, indices] = restored
            rgb = decoder.forward_from_coefficients(combined)["rgb"]
            if not bool(torch.isfinite(rgb).all()):
                raise ValueError("Nonfinite recovered RGB")
            return rgb  # No clipping; this does not authenticate KC.

    def _patient_logits(self, verified, km):
        key = permutation_key(km, verified.header.nm, PATIENT)
        decoder = self.model.watermark_decoder
        features = decoder.dct.watermark(verified.gray.to(self.device))
        restored = inverse_branch(features, key, verified.header.image_id, PATIENT, verified.raw.selectors)
        return decoder.forward_from_coefficients(restored)[0]

    def recover_patient(self, data, km, trusted_public_key):
        verified = self._verify(data, trusted_public_key)
        with fp32_inference(self.device):
            logits = self._patient_logits(verified, km)
            payload = logits_to_payload(logits)
        header = verified.header
        token = decrypt_patient(payload, km, header.nm, header.ngcm, verified.raw.h0bytes)
        return require_bytes(token, 16, "PatientToken")
