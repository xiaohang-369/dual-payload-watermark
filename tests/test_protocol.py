from dataclasses import replace
import hashlib
import hmac
import struct
from unittest.mock import patch
import numpy as np
import pytest
import torch
from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from dual_payload.crypto import (derive_key, COLOR_INFO, PATIENT_INFO, ENCRYPTION_INFO,
                                encrypt_patient, decrypt_patient, payload_to_bits, logits_to_payload)
from dual_payload.package import (Header, RawPackage, encode_package, decode_package, verify_package,
                                 gray_to_bytes, gray_from_bytes, FIXED_FIELDS)
from dual_payload.keyed_permutation import (COLOR, PATIENT, CHANNELS, COORDINATES, candidate, PRFStream,
    prf_context, fisher_yates, distortion, select_candidate, pack_selectors, unpack_selectors,
    permutation_key, permute_coefficients, inverse_branch)
from dual_payload.transforms import BlockDCT, WATERMARK_COORDS

KEY = bytes(range(32))
IMAGE = bytes(range(16))


@pytest.fixture
def header():
    return Header(IMAGE, bytes(range(32)), 1e6, 1e6, 2, 2, bytes(range(16)), bytes(range(16, 32)), bytes(range(12)))


@pytest.fixture
def signed(header):
    sk = Ed25519PrivateKey.from_private_bytes(KEY)
    gray = torch.linspace(-.2, 1.2, 65536).reshape(1, 1, 256, 256)
    return encode_package(header, bytes(range(256)) * 4, gray, sk), sk


def test_header_all_offsets_and_package_roundtrip(header, signed):
    data, sk = signed
    assert len(data) == 263384
    raw = decode_package(data)
    assert raw.encode() == data
    assert len(raw.h0bytes) == 152 and len(raw.selectors) == 1024 and len(raw.gbytes) == 262144
    assert data[:8] == b'DPWPKG01'
    assert data[8:12] == b'\x00\x01\x00\x98'
    assert data[12:28] == IMAGE
    assert data[28:40] == bytes.fromhex('010001000101010800010001')
    assert data[40:72] == bytes(range(32))
    assert data[72:74] == b'\x01\x0f'
    assert struct.unpack('>ddBB', data[74:92]) == (1e6, 1e6, 2, 2)
    assert data[92:108] == header.nc and data[108:124] == header.nm and data[124:136] == header.ngcm
    assert data[136:152] == bytes.fromhex('00010010001000100000010000010001')
    assert data[152:1176] == bytes(range(256)) * 4
    assert len(data[263320:]) == 64
    digest = hashlib.sha256(b'DPWSIG01' + data[:263320]).digest()
    sk.public_key().verify(data[263320:], digest)
    verified = verify_package(data, sk.public_key(), expected_model_id=header.model_id)
    assert verified.header == header
    assert gray_to_bytes(verified.gray) == raw.gbytes
    assert float(verified.gray.min()) < 0 and float(verified.gray.max()) > 1


@pytest.mark.parametrize('offset', [0, 10, 74, 151, 152, 1175, 1176, 263319, 263320, 263383])
def test_tamper_fails_signature_before_any_header_parsing(signed, header, offset):
    data, sk = signed
    tampered = bytearray(data); tampered[offset] ^= 1
    with patch.object(Header, 'decode', side_effect=AssertionError('must verify first')):
        with pytest.raises(InvalidSignature):
            verify_package(bytes(tampered), sk.public_key(), expected_model_id=header.model_id)


@pytest.mark.parametrize('name', list(FIXED_FIELDS))
def test_authenticated_invalid_fixed_fields_rejected(signed, header, name):
    data, sk = signed
    raw = decode_package(data)
    altered = bytearray(raw.h0bytes); altered[FIXED_FIELDS[name][0]] ^= 1
    prefix = bytes(altered) + raw.selectors + raw.gbytes
    signature = sk.sign(hashlib.sha256(b'DPWSIG01' + prefix).digest())
    with pytest.raises(ValueError, match=name):
        verify_package(prefix + signature, sk.public_key(), expected_model_id=header.model_id)


def test_package_length_model_id_and_trusted_key(signed):
    data, sk = signed
    for bad in (data[:-1], data + b'0'):
        with pytest.raises(ValueError, match='263384'):
            decode_package(bad)
    with pytest.raises(ValueError, match='ModelID'):
        verify_package(data, sk.public_key(), expected_model_id=bytes(32))
    with pytest.raises(InvalidSignature):
        verify_package(data, Ed25519PrivateKey.generate().public_key(), expected_model_id=bytes(32))


def test_signed_zero_and_nonfinite_rules(header, signed):
    header = replace(header, beta_c=-0.)
    assert header.encode()[74:82] == bytes(8)
    invalid = bytearray(header.encode()); invalid[74] = 0x80
    with pytest.raises(ValueError, match='Negative-zero'):
        Header.decode(bytes(invalid))
    gray = torch.zeros(1, 1, 256, 256); gray[0, 0, 0, 1] = -0.
    encoded = gray_to_bytes(gray)
    assert encoded[:8] == bytes.fromhex('0000000080000000')
    assert torch.signbit(gray_from_bytes(encoded)[0, 0, 0, 1])
    data, sk = signed
    raw = decode_package(data)
    gbytes = bytes.fromhex('7f800000') + raw.gbytes[4:]
    prefix = raw.h0bytes + raw.selectors + gbytes
    invalid = prefix + sk.sign(hashlib.sha256(b'DPWSIG01' + prefix).digest())
    with pytest.raises(ValueError, match='nonfinite'):
        verify_package(invalid, sk.public_key(), expected_model_id=raw.h0bytes[40:72])


@pytest.mark.parametrize('field,value', [('beta_c',float('nan')),('beta_m',float('inf')),('beta_c',-1.),
                                         ('min_moved_c',0),('min_moved_c',40),('min_moved_m',10)])
def test_invalid_header_parameters(header, field, value):
    with pytest.raises(ValueError):
        replace(header, **{field:value}).encode()


@pytest.mark.parametrize('info', [COLOR_INFO, PATIENT_INFO, ENCRYPTION_INFO])
def test_hkdf_matches_independent_library(info):
    expected = HKDF(algorithm=hashes.SHA256(), length=32, salt=IMAGE, info=info).derive(KEY)
    assert derive_key(KEY, IMAGE, info) == expected


def test_payload_authentication_and_msb_bit_order(header):
    token = bytes(range(16))
    payload = encrypt_patient(token, KEY, header.nm, header.ngcm, header.encode())
    assert len(payload) == 32
    assert decrypt_patient(payload, KEY, header.nm, header.ngcm, header.encode()) == token
    bits = payload_to_bits(bytes([0x81,0x7e]) + bytes(30))
    assert bits[:16].tolist() == [1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,0]
    assert logits_to_payload(bits * 2 - 1) == bytes([0x81,0x7e]) + bytes(30)
    assert logits_to_payload(torch.zeros(256)) == bytes([255])*32
    for offset in (0, 15, 16, 31):
        bad = bytearray(payload); bad[offset] ^= 1
        with pytest.raises(InvalidTag):
            decrypt_patient(bytes(bad), KEY, header.nm, header.ngcm, header.encode())
    with pytest.raises(InvalidTag):
        decrypt_patient(payload, bytes(32), header.nm, header.ngcm, header.encode())
    with pytest.raises(InvalidTag):
        decrypt_patient(payload, KEY, header.nm, header.ngcm, bytes(152))
    for bad in (torch.zeros(255), torch.full((256,),float('nan'))):
        with pytest.raises(ValueError):
            logits_to_payload(bad)


def test_prf_context_counter_consumption_and_overflow():
    context = prf_context(IMAGE, COLOR, 2, 31, 7, 15)
    assert context == b'DPWPERM1' + IMAGE + bytes.fromhex('010002001f070f')
    assert len(context) == 31
    expected = b''.join(hmac.digest(KEY, context + i.to_bytes(4,'big'),'sha256') for i in range(3))
    stream = PRFStream(KEY, context)
    assert bytes(stream.byte() for _ in range(96)) == expected
    stream.counter = 2**32
    with pytest.raises(ValueError, match='exhausted'):
        stream.byte()


def test_fisher_yates_rejection_and_q_256():
    class Stream:
        def __init__(self, data): self.data=iter(data); self.calls=0
        def byte(self): self.calls+=1; return next(self.data)
    stream = Stream([255, 254, 255])  # L=3 rejects 255; L=2 accepts 255 (Q=256).
    assert fisher_yates(3, stream) == (0,1,2)
    assert stream.calls == 3
    stream = Stream([255] * 20)
    # n=2: Q must not overflow to zero.
    assert fisher_yates(2, stream) == (0,1)


def test_layer_orders_and_determinism_and_branch_domains():
    assert COORDINATES[PATIENT] == WATERMARK_COORDS
    assert len(CHANNELS[COLOR]) == 39 and len(CHANNELS[PATIENT]) == 9
    ckey = permutation_key(KEY, IMAGE, COLOR)
    mkey = permutation_key(KEY, IMAGE, PATIENT)
    assert ckey != mkey
    for branch, key in ((COLOR, ckey),(PATIENT,mkey)):
        pi = candidate(key,IMAGE,branch,0,0,1)
        assert pi == candidate(key,IMAGE,branch,0,0,1)
        assert sorted(pi) == list(range(len(CHANNELS[branch])))
        assert any(pi != candidate(key,IMAGE,branch,0,0,r) for r in range(2,16))
        coords = COORDINATES[branch]
        assert all(sum(coords[i]) == sum(coords[p]) for i,p in enumerate(pi))
    assert derive_key(KEY,IMAGE,PATIENT_INFO) != derive_key(KEY,IMAGE,ENCRYPTION_INFO)


def test_selectors_all_nibbles():
    color = tuple(i % 16 for i in range(1024)); patient = tuple((i//16)%16 for i in range(1024))
    packed = pack_selectors(color, patient)
    assert len(packed)==1024 and packed[1]==0x10 and packed[16]==0x01
    assert unpack_selectors(packed)==(color,patient)


def test_selection_ties_duplicates_structural_filter_and_beta():
    identity = tuple(range(9)); swap=(1,0,*range(2,9))
    larger = (2,1,0,*range(3,9))
    def candidates(key,image,branch,row,col,r): return identity if r==1 else swap if r in (2,3) else larger
    values = np.arange(9,dtype=np.float32)
    with patch('dual_payload.keyed_permutation.candidate',side_effect=candidates) as gen:
        assert select_candidate(values,KEY,IMAGE,PATIENT,0,0,2,2)[0]==2
        assert gen.call_count == 15  # Keep duplicate r=3; no resampling.
        assert select_candidate(values,KEY,IMAGE,PATIENT,0,0,2,1.99)[0]==0
        assert select_candidate(values,KEY,IMAGE,PATIENT,0,0,2,100)[0]==2
        assert select_candidate(values,KEY,IMAGE,PATIENT,0,0,3,100)[0]==0
        assert select_candidate(values,KEY,IMAGE,PATIENT,0,0,1,100)[0]==2
        constant=np.ones(9,dtype=np.float32)
        assert select_candidate(constant,KEY,IMAGE,PATIENT,0,0,2,0)[0]==2
    # binary64 can retain FP32-scale squared differences exceeding FP32 maximum.
    extreme=np.array([3e30,-3e30]+[0]*7,dtype=np.float32)
    assert np.isfinite(distortion(extreme,swap)) and distortion(extreme,swap)>1e60


def test_full_coefficient_inverse_independence_and_float_roundtrip(header):
    source = torch.randn(1,64,32,32)
    ckey,mkey=permutation_key(KEY,header.nc,COLOR),permutation_key(KEY,header.nm,PATIENT)
    transformed,selectors=permute_coefficients(source,ckey,mkey,header)
    structure=[i for i in range(64) if i not in CHANNELS[COLOR]+CHANNELS[PATIENT]]
    assert torch.equal(transformed[:,structure],source[:,structure])
    for branch,key in ((COLOR,ckey),(PATIENT,mkey)):
        idx=list(CHANNELS[branch])
        restored=inverse_branch(transformed[:,idx],key,IMAGE,branch,selectors)
        assert torch.equal(restored,source[:,idx])
        # Changing the other branch's nibble cannot affect this restoration.
        altered=bytes(x ^ (0x0f if branch==COLOR else 0xf0) for x in selectors)
        assert torch.equal(restored,inverse_branch(transformed[:,idx],key,IMAGE,branch,altered))
    dct=BlockDCT(); roundtrip=dct(dct.inverse(transformed))
    torch.testing.assert_close(roundtrip,transformed,atol=3e-6,rtol=2e-6)
    assert not torch.equal(roundtrip,transformed)


def test_nonfinite_coefficients_do_not_become_identity_fallback():
    for invalid in (float('nan'),float('inf'),-float('inf')):
        values=np.zeros(9,dtype=np.float32);values[0]=invalid
        with pytest.raises(ValueError,match='finite FP32'):
            select_candidate(values,KEY,IMAGE,PATIENT,0,0,2,1.)


def test_selector_receiver_does_not_rescore_candidates():
    features=torch.randn(1,9,32,32)
    # All r=15, irrespective of any sending beta/min_moved choices.
    selectors=bytes([15])*1024
    with patch('dual_payload.keyed_permutation.select_candidate',side_effect=AssertionError('no receiver selection')):
        restored=inverse_branch(features,KEY,IMAGE,PATIENT,selectors)
    pi=candidate(KEY,IMAGE,PATIENT,0,0,15)
    assert torch.equal(restored[0,list(pi),0,0],features[0,:,0,0])


def test_beta_comparison_uses_encoded_binary64_value(header):
    values=np.zeros(9,dtype=np.float32);values[1]=2**27
    swap=(1,0,*range(2,9))
    beta=2**55-1  # This integer rounds to 2**55 in the header's binary64 field.
    encoded=replace(header,beta_m=beta).encode()
    assert struct.unpack_from('>d',encoded,82)[0]==2**55
    with patch('dual_payload.keyed_permutation.candidate',return_value=swap):
        assert select_candidate(values,KEY,IMAGE,PATIENT,0,0,2,beta)[0]==1
