"""PAD patient-group splits, Derm7pt external cohorts, and common preprocessing."""
import csv
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .artifacts import load_record, save_record, sha256
from .preprocess import integer_pixels, prepare_work_image
from .profile import FIXED

SOURCES = {
    'pad': 'https://data.mendeley.com/datasets/zr7vgbcyr2/1',
    'derm7pt': 'https://github.com/jeremykawahara/derm7pt',
}


def read_rows(path, required):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Missing or duplicate CSV columns')
        if not set(required) <= set(reader.fieldnames):
            raise ValueError(f'Missing columns: {set(required) - set(reader.fieldnames)}')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty metadata')
    return rows


def present(value):
    return isinstance(value, str) and value.strip().lower() not in ('', 'nan', 'none', 'null', 'na')


def safe_path(root, relative):
    root = Path(root).resolve()
    if not present(relative) or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Image path must be relative to its dataset root')
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Missing or unsafe image path: {relative}')
    return path


def image_record(root, relative, **fields):
    path = safe_path(root, relative)
    rgb, rect = prepare_work_image(path)  # Same rules as sender, not the old crop loader.
    return dict(fields, path=path.relative_to(Path(root).resolve()).as_posix(),
                file_sha256=sha256(path),
                work_sha256=hashlib.sha256(integer_pixels(rgb).tobytes()).hexdigest(),
                content_rect=list(rect))


def pad_records(root, metadata):
    rows = read_rows(metadata, ('patient_id', 'lesion_id', 'img_id', 'diagnostic'))
    # Releases can be unpacked into multiple image subdirectories. Resolve basenames uniquely.
    files = defaultdict(list)
    for path in Path(root).rglob('*.png'):
        files[path.name].append(path)
    records, seen = [], set()
    for row in rows:
        if not all(present(row[k]) for k in ('patient_id', 'lesion_id', 'img_id', 'diagnostic')):
            raise ValueError('PAD identifying fields must not be missing')
        image_id = row['img_id'].strip()
        name = image_id if image_id.endswith('.png') else image_id + '.png'
        if name in seen or len(files[name]) != 1:
            raise ValueError(f'Duplicate metadata or missing/ambiguous PAD image: {name}')
        seen.add(name)
        relative = files[name][0].relative_to(root).as_posix()
        records.append(image_record(root, relative, dataset='pad', image_id=image_id,
                                    patient_id=row['patient_id'].strip(), case_id=row['lesion_id'].strip(),
                                    diagnosis=row['diagnostic'].strip(), modality='clinical'))
    return records


def derm_records(root, metadata):
    rows = read_rows(metadata, ('case_num', 'clinic', 'derm', 'diagnosis'))
    records, seen, missing = [], set(), []
    original_splits = {}
    split_files = [Path(metadata).parent / (name + '_indexes.csv') for name in ('train', 'valid', 'test')]
    if any(p.exists() for p in split_files):
        if not all(p.exists() for p in split_files):
            raise ValueError('Incomplete official Derm7pt split index files')
        for name, path in zip(('train', 'valid', 'test'), split_files):
            for index in read_rows(path, ('indexes',)):
                number = int(index['indexes'])
                if number in original_splits or not 0 <= number < len(rows):
                    raise ValueError('Overlapping or out-of-range Derm7pt row indexes')
                original_splits[number] = name
        if len(original_splits) != len(rows):
            raise ValueError('Official Derm7pt indexes do not cover all cases')
    for index, row in enumerate(rows):
        case = row['case_num'].strip()
        if not present(case) or case in seen:
            raise ValueError('Missing or repeated Derm7pt case number')
        seen.add(case)
        for column, modality in (('clinic', 'clinical'), ('derm', 'dermoscopic')):
            if not present(row[column]):
                missing.append({'case_id': case, 'modality': modality, 'reason': 'metadata_missing'})
                continue
            relative = 'images/' + row[column].strip()
            records.append(image_record(root, relative, dataset='derm7pt', image_id=case + ':' + column,
                                        patient_id=None, case_id=case, diagnosis=row['diagnosis'].strip(),
                                        modality=modality, split='external',
                                        original_split=original_splits.get(index, 'unspecified')))
    if not records:
        raise ValueError('No Derm7pt images')
    return records, missing


def assign_pad_splits(records, fractions, seed):
    if len(fractions) != 3 or any(not math.isfinite(x) or x <= 0 for x in fractions) or not math.isclose(sum(fractions), 1):
        raise ValueError('Three positive split fractions must sum to one')
    # Merge patients connected by exact duplicate work images BEFORE splitting.
    parent = {r['patient_id']: r['patient_id'] for r in records}
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    owners = {}
    for row in records:
        patient = row['patient_id']
        for key in (('file', row['file_sha256']), ('work', row['work_sha256']), ('lesion', row['case_id'])):
            if key in owners:
                a, b = find(patient), find(owners[key])
                parent[max(a, b)] = min(a, b)
            owners[key] = patient
    groups = sorted({find(p) for p in parent}, key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).digest())
    if len(groups) < 3:
        raise ValueError('At least three independent PAD groups are required')
    n_train = max(1, min(len(groups) - 2, round(len(groups) * fractions[0])))
    n_valid = max(1, min(len(groups) - n_train - 1, round(len(groups) * fractions[1])))
    splits = {g: 'train' if i < n_train else 'valid' if i < n_train + n_valid else 'test'
              for i, g in enumerate(groups)}
    for row in records:
        row['group_id'] = find(row['patient_id'])
        row['split'] = splits[row['group_id']]


def audit_records(records):
    identities, ownership = set(), {}
    for row in records:
        identity = (row['dataset'], row['image_id'])
        if identity in identities:
            raise ValueError('Duplicate manifest image identity')
        identities.add(identity)
        if row['dataset'] == 'derm7pt' and row['split'] != 'external':
            raise ValueError('Derm7pt is reserved for external evaluation')
        if row['dataset'] == 'pad' and row['split'] not in ('train', 'valid', 'test'):
            raise ValueError('Invalid PAD split')
        keys = [('case', row['dataset'], row['case_id']),
                ('file', row['file_sha256']), ('work', row['work_sha256'])]
        if row.get('patient_id'):
            keys.append(('patient', row['dataset'], row['patient_id']))
        for key in keys:
            if key in ownership and ownership[key] != row['split']:
                raise ValueError(f'Cross-split patient/case or exact duplicate leakage: {key[0]}')
            ownership[key] = row['split']


def create_manifest(pad_root, pad_metadata, output, fractions, seed, derm_root=None, derm_metadata=None):
    pad_root = Path(pad_root).resolve()
    records = pad_records(pad_root, pad_metadata)
    assign_pad_splits(records, fractions, seed)
    sources = {'pad': {'url': SOURCES['pad'], 'metadata_sha256': sha256(pad_metadata)}}
    missing = []
    if (derm_root is None) != (derm_metadata is None):
        raise ValueError('Provide both Derm7pt root and metadata')
    if derm_root is not None:
        external, missing = derm_records(Path(derm_root).resolve(), derm_metadata)
        records.extend(external)
        sources['derm7pt'] = {'url': SOURCES['derm7pt'], 'metadata_sha256': sha256(derm_metadata)}
    audit_records(records)
    counts = Counter(f"{r['dataset']}/{r['split']}/{r['modality']}" for r in records)
    diagnoses = Counter(f"{r['dataset']}/{r['split']}/{r['diagnosis']}" for r in records)
    return save_record(output, {'schema': 'medical-manifest-v1', 'sources': sources,
                               'seed': seed, 'pad_group_fractions': list(fractions),
                               'preprocess': FIXED['preprocess'], 'counts': dict(counts),
                               'diagnoses': dict(diagnoses), 'missing_views': missing,
                               'duplicate_check': 'exact source bytes and 256x256 work pixels; near duplicates not certified',
                               'records': sorted(records, key=lambda r: (r['dataset'], r['image_id']))})


class MedicalDataset(Dataset):
    def __init__(self, manifest, roots, split, modality=None):
        self.manifest = load_record(manifest)
        if self.manifest['schema'] != 'medical-manifest-v1' or self.manifest['preprocess'] != FIXED['preprocess']:
            raise ValueError('Unsupported data manifest or preprocessing')
        if split not in ('train', 'valid', 'test', 'external'):
            raise ValueError('Unknown split')
        audit_records(self.manifest['records'])
        self.records = [r for r in self.manifest['records'] if r['split'] == split and
                        (modality is None or r['modality'] == modality)]
        self.roots, self.split = roots, split
        if not self.records:
            raise ValueError(f'Empty split: {split}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = safe_path(self.roots[record['dataset']], record['path'])
        if sha256(path) != record['file_sha256']:
            raise ValueError('Source image changed after manifest creation')
        rgb, rect = prepare_work_image(path)
        if list(rect) != record['content_rect'] or hashlib.sha256(integer_pixels(rgb).tobytes()).hexdigest() != record['work_sha256']:
            raise ValueError('Working image changed after manifest creation')
        return {'rgb': rgb[0], 'content_rect': torch.tensor(rect), 'index': index,
                'image_id': record['dataset'] + ':' + record['image_id'], 'modality': record['modality']}
