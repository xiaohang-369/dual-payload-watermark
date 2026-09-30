import hashlib

import pytest

from dual_payload.medical.profile import load_profile, register_profile, template


@pytest.fixture(scope="session")
def test_config():
    config = template()
    config.update(profile_id=65000, quantization_id=65000, purpose="test",
                  quantization_steps=[0.025 + i * 0.001 for i in range(39)],
                  rms_limits={"color_residual": 2 / 255, "color_ciphertext": 2 / 255,
                              "patient_ciphertext": 2 / 255},
                  weights={name: hashlib.sha256(name.encode()).hexdigest()
                           for name in ("ec", "ew", "dc", "dw")})
    return config


@pytest.fixture(scope="session")
def profile(test_config, tmp_path_factory):
    path = register_profile(test_config, tmp_path_factory.mktemp("profiles"), allow_test=True)
    return load_profile(path, allow_test=True)
