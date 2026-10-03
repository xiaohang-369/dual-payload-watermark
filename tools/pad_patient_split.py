"""Deterministic multi-diagnostic patient grouping; no image processing or training."""
from collections import defaultdict
import math
import random

import numpy as np

DIAGNOSTICS = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
SPLITS = ("train", "val", "test")
RATIOS = (.70, .15, .15)
PATIENT_TARGETS = (961, 206, 206)
SEED = 2026
FEATURES = ("image_count", "lesion_count", *(
    f"{diagnostic}_{unit}" for unit in ("image_count", "lesion_count", "patient_count")
    for diagnostic in DIAGNOSTICS))


def build_patient_vectors(rows):
    groups = defaultdict(list)
    seen_images, lesion_diagnoses = set(), {}
    for row in rows:
        for field in ("patient_id", "lesion_id", "img_id", "diagnostic"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing {field}")
        if row["diagnostic"] not in DIAGNOSTICS:
            raise ValueError(f"Unknown diagnostic: {row['diagnostic']}")
        if row["img_id"] in seen_images:
            raise ValueError(f"Duplicate img_id: {row['img_id']}")
        seen_images.add(row["img_id"])
        key = (row["patient_id"], row["lesion_id"])
        if key in lesion_diagnoses and lesion_diagnoses[key] != row["diagnostic"]:
            raise ValueError(f"Conflicting lesion_key diagnostic: {key}")
        lesion_diagnoses[key] = row["diagnostic"]
        groups[row["patient_id"]].append(row)
    ids = sorted(groups)
    vectors = []
    for patient in ids:
        group = groups[patient]
        by_diag = [[r for r in group if r["diagnostic"] == d] for d in DIAGNOSTICS]
        vectors.append([len(group), len({r["lesion_id"] for r in group}),
                        *[len(g) for g in by_diag],
                        *[len({r["lesion_id"] for r in g}) for g in by_diag],
                        *[int(bool(g)) for g in by_diag]])
    return ids, np.asarray(vectors, dtype=np.int64).reshape(len(ids), 20)


def split_patients(rows, patient_targets=PATIENT_TARGETS, ratios=RATIOS, seed=SEED, max_passes=30):
    if max_passes < 1:
        raise ValueError("max_passes must be positive")
    ids, vectors = build_patient_vectors(rows)
    quotas = np.asarray(patient_targets, dtype=np.int64)
    ratios = np.asarray(ratios, dtype=np.float64)
    if (quotas.shape != (3,) or np.any(quotas <= 0) or quotas.sum() != len(ids)
            or ratios.shape != (3,) or np.any(ratios <= 0) or not np.isfinite(ratios).all()
            or not math.isclose(float(ratios.sum()), 1)):
        raise ValueError("Invalid patient quotas or split ratios")
    totals = vectors.sum(axis=0)
    mel_column = 14 + DIAGNOSTICS.index("MEL")
    if totals[mel_column] < 3:
        raise ValueError("MEL requires at least three patients")
    if np.any(totals[14:] < 3):
        raise ValueError("Every diagnostic requires at least three patients")
    targets = ratios[:, None] * totals
    denominator = np.where(targets > 0, targets, 1)
    weights = 1 / denominator ** 2
    rng = random.Random(seed)
    tie_order = list(range(len(ids)))
    rng.shuffle(tie_order)
    tie_rank = {patient: rank for rank, patient in enumerate(tie_order)}
    rarity = (vectors[:, 14:] / totals[14:]).sum(axis=1)
    order = sorted(range(len(ids)), key=lambda i: (-rarity[i], -vectors[i, 0], tie_rank[i], ids[i]))
    assignment = np.full(len(ids), -1, dtype=np.int64)
    counts = np.zeros((3, 20), dtype=np.int64)
    sizes = np.zeros(3, dtype=np.int64)

    mel_total = int(totals[mel_column])
    mel_targets = ratios * mel_total
    mel_quotas = np.maximum(1, np.floor(mel_targets).astype(int))
    while mel_quotas.sum() < mel_total:
        options = [s for s in range(3) if mel_quotas[s] < quotas[s]]
        if not options:
            raise ValueError("MEL allocation exceeds patient capacities")
        chosen = max(options, key=lambda s: (mel_targets[s] - mel_quotas[s], -s))
        mel_quotas[chosen] += 1
    while mel_quotas.sum() > mel_total:
        chosen = max((s for s in range(3) if mel_quotas[s] > 1),
                     key=lambda s: (mel_quotas[s] - mel_targets[s], -s))
        mel_quotas[chosen] -= 1
    if np.any(mel_quotas > quotas):
        raise ValueError("MEL quotas exceed patient capacity")
    lower = np.maximum(1, np.floor(mel_targets).astype(int))
    upper = np.maximum(1, np.ceil(mel_targets).astype(int))
    if lower.sum() > mel_total:
        lower[:] = 1  # Small synthetic cohorts still require presence in all groups.

    for mel_phase in (True, False):
        for i in order:
            if bool(vectors[i, mel_column]) != mel_phase:
                continue
            options = [s for s in range(3) if sizes[s] < quotas[s]
                       and (not mel_phase or counts[s, mel_column] < mel_quotas[s])]
            if not options:
                raise ValueError("No feasible patient allocation")
            delta = ((2 * (counts - targets) * vectors[i] + vectors[i] ** 2) * weights).sum(axis=1)
            s = min(options, key=lambda s: (delta[s], s))
            assignment[i] = s
            counts[s] += vectors[i]
            sizes[s] += 1
    if np.any(counts[:, 14:] == 0):
        raise ValueError("Greedy initialization does not cover all six diagnostics; no output published")
    initial = float(((counts - targets) ** 2 * weights).sum())
    swaps = 0
    converged = False
    for pass_index in range(max_passes):
        pass_swaps = 0
        visit_order = list(range(len(ids)))
        rng.shuffle(visit_order)
        for i in visit_order:
            a = int(assignment[i])
            best = None
            for b in range(3):
                if b == a:
                    continue
                candidates = np.flatnonzero(assignment == b)
                delta = vectors[candidates] - vectors[i]
                new_a = counts[a] + delta
                new_b = counts[b] - delta
                feasible = ((new_a[:, 14:] > 0).all(axis=1) & (new_b[:, 14:] > 0).all(axis=1)
                            & (new_a[:, mel_column] >= lower[a]) & (new_a[:, mel_column] <= upper[a])
                            & (new_b[:, mel_column] >= lower[b]) & (new_b[:, mel_column] <= upper[b]))
                change = (2 * delta * ((counts[a] - targets[a]) * weights[a]
                                      - (counts[b] - targets[b]) * weights[b])
                          + delta ** 2 * (weights[a] + weights[b])).sum(axis=1)
                change[~feasible] = np.inf
                at = int(np.argmin(change))
                proposal = (float(change[at]), int(candidates[at]), b)
                if proposal[0] < -1e-12 and (best is None or proposal < best):
                    best = proposal
            if best is not None:
                _, j, b = best
                delta = vectors[j] - vectors[i]
                counts[a] += delta
                counts[b] -= delta
                assignment[i], assignment[j] = b, a
                pass_swaps += 1
        swaps += pass_swaps
        if pass_swaps == 0:
            converged = True
            break
    if not np.array_equal(np.bincount(assignment, minlength=3), quotas):
        raise ValueError("Patient quota invariant failed")
    if np.any(counts[:, mel_column] < lower) or np.any(counts[:, mel_column] > upper):
        raise ValueError("MEL presence outside floor/ceil targets")
    evidence = {"algorithm": "rare-first greedy with MEL allocation and seeded improving patient swaps",
                "seed": seed, "feature_order": list(FEATURES), "dimension_weights": "equal",
                "objective": "sum(((actual-target)/where(target>0,target,1))**2)",
                "initial_objective": initial,
                "final_objective": float(((counts - targets) ** 2 * weights).sum()),
                "max_passes": max_passes, "passes": pass_index + 1, "accepted_swaps": swaps,
                "stop_reason": "no_improving_swap" if converged else "pass_limit",
                "mel_patient_bounds": {s: [int(lower[i]), int(upper[i])] for i, s in enumerate(SPLITS)},
                "global_optimum_claimed": False}
    return {patient: SPLITS[int(assignment[i])] for i, patient in enumerate(ids)}, evidence
