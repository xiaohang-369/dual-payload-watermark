"""Sionna 2.1.0 paired 5G NR LDPC, followed by the registered tensor layout."""

from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch

from .profile import Profile
from .protocol import BLOCKS, CHANNELS, FRAME_BYTES, Branch, bits_to_bytes, bytes_to_bits


class TransportCodec:
    def __init__(self, profile: Profile, device="cpu", decode_batch_size=8):
        try:
            installed_version = version("sionna")
        except PackageNotFoundError as exc:
            raise RuntimeError("Medical V1 requires sionna==2.1.0; install .[medical]") from exc
        if installed_version != "2.1.0":
            raise RuntimeError("Medical V1 requires sionna==2.1.0; install .[medical]")
        from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder

        if type(decode_batch_size) is not int or decode_batch_size < 1:
            raise ValueError("decode_batch_size must be positive")
        self.profile, self.device, self.batch_size = profile, device, decode_batch_size
        self.encoder = LDPC5GEncoder(1024, 1536, bg="bg2", num_bits_per_symbol=None,
                                     precision="single", device=device)
        self.decoder = LDPC5GDecoder(self.encoder, cn_update="boxplus-phi",
                                     cn_schedule="flooding", hard_out=True, return_infobits=True,
                                     num_iter=20, llr_max=20, harq_mode=False,
                                     precision="single", device=device)

    @torch.no_grad()
    def encode(self, frame: bytes, branch: Branch) -> torch.Tensor:
        if len(frame) != FRAME_BYTES[branch]:
            raise ValueError("Wrong frame size")
        padded = frame + bytes(BLOCKS[branch] * 128 - len(frame))
        bits = torch.tensor(bytes_to_bits(padded).reshape(-1, 1024),
                            dtype=torch.float32, device=self.device)
        coded = self.encoder(bits).flatten()
        permutation = torch.tensor(self.profile.permutation(branch).copy(), device=self.device)
        laid_out = torch.cat((coded[permutation], coded.new_zeros(512)))
        return laid_out.reshape(1, CHANNELS[branch], 32, 32)

    @torch.no_grad()
    def decode_information(self, logits: torch.Tensor, branch: Branch) -> np.ndarray:
        if logits.shape != (1, CHANNELS[branch], 32, 32) or not torch.isfinite(logits).all():
            raise ValueError("Invalid branch logits")
        # Discard layout padding before inverse permutation; do not threshold logits.
        useful = logits.detach().to(device=self.device, dtype=torch.float32).flatten()[:-512]
        inverse = torch.tensor(self.profile.permutation(branch, inverse=True).copy(), device=self.device)
        blocks = useful[inverse].reshape(-1, 1536)
        decoded = torch.cat([self.decoder(part) for part in blocks.split(self.batch_size)])
        return decoded.cpu().numpy().astype(np.uint8)

    @staticmethod
    def information_frame(information: np.ndarray, branch: Branch) -> bytes:
        raw = bits_to_bytes(information.reshape(-1))
        length = FRAME_BYTES[branch]
        if any(raw[length:]):
            raise ValueError("Nonzero LDPC data padding")
        return raw[:length]

    def decode(self, logits: torch.Tensor, branch: Branch) -> bytes:
        return self.information_frame(self.decode_information(logits, branch), branch)
