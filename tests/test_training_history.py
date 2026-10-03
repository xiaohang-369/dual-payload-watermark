"""Training engineering checks using temporary synthetic images only."""
from copy import deepcopy
import json
from unittest.mock import patch

from PIL import Image
import pytest
import torch

from dual_payload import training
from dual_payload.checkpoints import load_checkpoint, model_from_file
from dual_payload.config import DEFAULT_CONFIG
from dual_payload.data import fixed_message
from dual_payload.protocol import ProtocolV1


LOSS_KEYS = {"rgb", "chroma", "luma", "message", "carrier", "range", "total"}
METRIC_KEYS = {"carrier_psnr", "carrier_ssim", "rgb_psnr", "rgb_ssim",
               "rgb_psnr_clipped", "rgb_ssim_clipped", "ber", "bit_accuracy",
               "message_accuracy", "carrier_oob_fraction", "rgb_oob_fraction",
               "delta_c_rms", "delta_w_rms"}


def manifest(tmp_path, count=13, name="train", explicit=False):
    samples = []
    for index in range(count):
        image = tmp_path / f"{name}-图像-{index}.png"
        Image.new("RGB", (256, 256), (index, index + 1, index + 2)).save(image)
        row = {"path": image.name, "patient_id": f"{name}-p{index}", "img_id": image.name}
        if explicit:
            row["message"] = [index % 2] * 256
        samples.append(row)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"samples": samples}, ensure_ascii=False))
    return path


def config_file(tmp_path, train=None, val=None, epochs=1, batch_size=2):
    config = deepcopy(DEFAULT_CONFIG)
    config["device"] = "cpu"
    config["data"].update(train_manifest=str(train) if train else None,
                          val_manifest=str(val) if val else None, batch_size=batch_size)
    config["train"].update(epochs=epochs, output_dir=str(tmp_path / "run"))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path, config


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_jsonl_rejects_nonfinite_without_partial_append(tmp_path, invalid):
    path = tmp_path / "metrics.jsonl"
    record = {"validation": {"说明": "逐轮记录", "ber": .5}}
    training.append_jsonl(path, record)
    before = path.read_bytes()
    assert "逐轮记录" in path.read_text()
    assert json.loads(path.read_text()) == record
    with pytest.raises(ValueError):
        training.append_jsonl(path, {"train_loss": {"total": invalid}})
    assert path.read_bytes() == before


def test_jsonl_flushes_and_fsyncs_each_record(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with patch("dual_payload.training.os.fsync", wraps=training.os.fsync) as sync:
        training.append_jsonl(path, {"epoch": 0})
        assert path.read_text().splitlines() == ['{"epoch": 0}']
        assert sync.call_count == 1
        training.append_jsonl(path, {"epoch": 1})
        assert len(path.read_text().splitlines()) == 2
        assert sync.call_count == 2


def test_two_epoch_history_sample_weighting_and_live_append(tmp_path, monkeypatch, capsys):
    path, _ = config_file(tmp_path, epochs=2)
    run = tmp_path / "run"
    make_dataset = training.SyntheticDataset
    monkeypatch.setattr(training, "SyntheticDataset", lambda count, **kwargs:
                        make_dataset(3 if kwargs.get("training") else 1, **kwargs))
    calls = []

    def step(model, batch, *args):
        # Uneven batches of 2 and 1; deliberately different batch losses.
        epoch = len(calls) // 2
        if epoch == 1:
            assert len((run / "metrics.jsonl").read_text().splitlines()) == 1
        count = len(batch["rgb"])
        calls.append(count)
        base = (1 if count == 2 else 10) + 2 * epoch
        return {key: float(base + i) for i, key in enumerate(sorted(LOSS_KEYS))}

    validations = [{"ber": .5, "loss_total": 4., "extra_metric_原样保留": 7.},
                   {"ber": .4, "loss_total": 3., "extra_metric_原样保留": 8.}]
    validation_calls = []

    def validate(*args):
        value = validations[len(validation_calls)]
        validation_calls.append(value)
        return value

    monkeypatch.setattr(training, "train_step", step)
    monkeypatch.setattr(training, "validate", validate)
    monkeypatch.setattr(training, "save_checkpoint", lambda *args, **kwargs: None)
    training.train_main(["--config", str(path), "--smoke"])
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 2 and calls == [2, 1, 2, 1]
    for epoch, record in enumerate(records):
        assert set(record) == {"epoch", "global_step", "train_samples", "train_loss", "validation", "mode"}
        assert record["epoch"] == epoch and record["global_step"] == (epoch + 1) * 2
        assert record["train_samples"] == 3 and record["mode"] == "joint_256"
        assert record["validation"] == validations[epoch]
        for i, key in enumerate(sorted(LOSS_KEYS)):
            assert record["train_loss"][key] == pytest.approx(4 + 2 * epoch + i)
    assert json.loads((run / "validation.json").read_text()) == validations[-1]
    assert '"step": 1' in capsys.readouterr().out
    assert not (run / "overfit8_selection.json").exists()


def test_existing_best_last_conditions_and_epoch_numbering(tmp_path, monkeypatch):
    path, _ = config_file(tmp_path, epochs=4)
    monkeypatch.setattr(training, "train_step", lambda *args: {key: 1. for key in LOSS_KEYS})
    metrics = iter([{"ber": .5, "loss_total": 4.}, {"ber": .25, "loss_total": 5.},
                    {"ber": .25, "loss_total": 3.}, {"ber": .3, "loss_total": 3.}])
    monkeypatch.setattr(training, "validate", lambda *args: next(metrics))
    saved = []

    def save(path, model, config, **state):
        saved.append((path.name, state["epoch"]))
        assert state["overfit8"] is False

    monkeypatch.setattr(training, "save_checkpoint", save)
    training.train_main(["--config", str(path), "--smoke"])
    assert [epoch for name, epoch in saved if name == "last.pt"] == [0, 1, 2, 3]
    assert [epoch for name, epoch in saved if name == "best_message_ber.pt"] == [0, 1]
    assert [epoch for name, epoch in saved if name == "best.pt"] == [0, 2]
    records = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in records] == [0, 1, 2, 3]


def test_overfit_selection_and_pairs_stable_across_rebuilds_and_epochs(tmp_path):
    path = manifest(tmp_path)
    before = path.read_bytes()
    data, selected = training.make_overfit8_dataset(path, 2026)
    rebuilt, selected_again = training.make_overfit8_dataset(path, 2026)
    assert selected == selected_again and data.indices == rebuilt.indices
    assert len(data) == len(set(data.indices)) == 8
    assert data.indices == sorted(data.indices)
    assert data.dataset.training is False
    _, config = config_file(tmp_path)
    expected_pairs = {}
    for position, original in enumerate(data.indices):
        first = data[position]
        row = selected["samples"][position]
        assert row["original_manifest_row_index"] == original
        assert row["path"] == data.dataset.rows[original]["path"]
        assert row["patient_id"] == data.dataset.rows[original]["patient_id"]
        assert row["img_id"] == data.dataset.rows[original]["img_id"]
        assert torch.equal(first["message"], fixed_message(original, 2026))
        assert torch.equal(first["message"], data[position]["message"])
        assert torch.equal(first["message"], rebuilt[position]["message"])
        assert first["message"].dtype == torch.float32 and first["message"].shape == (256,)
        assert ((first["message"] == 0) | (first["message"] == 1)).all()
        expected_pairs[first["path"]] = first["message"]
    for epoch in (0, 1, 5):
        for batch in training.make_loader(data, config, training=True, epoch=epoch):
            for sample_path, message in zip(batch["path"], batch["message"]):
                assert torch.equal(message, expected_pairs[sample_path])
    assert data.dataset.manifest.read_bytes() == before


def test_overfit_reuses_explicit_manifest_messages(tmp_path):
    path = manifest(tmp_path, explicit=True)
    data, _ = training.make_overfit8_dataset(path, 2026)
    for i, original in enumerate(data.indices):
        assert torch.equal(data[i]["message"], torch.tensor(data.dataset.rows[original]["message"], dtype=torch.float32))


@pytest.mark.parametrize("count", [1, 7])
def test_overfit_rejects_too_few_samples(tmp_path, count):
    path = manifest(tmp_path, count=count)
    with pytest.raises(ValueError, match="at least 8"):
        training.make_overfit8_dataset(path, 2026)


def test_overfit_requires_train_manifest(tmp_path):
    path, _ = config_file(tmp_path)
    with pytest.raises(ValueError, match="train_manifest"):
        training.train_main(["--config", str(path), "--overfit8"])
    assert not (tmp_path / "run").exists()


def test_overfit_and_synthetic_smoke_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit) as error:
        training.train_main(["--config", "unused.json", "--overfit8", "--smoke"])
    assert error.value.code == 2


@pytest.fixture(scope="module")
def overfit_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-overfit8")
    manifest_path = manifest(root)
    path, config = config_file(root, train=manifest_path, epochs=2)
    run = root / "run"
    loaders = []
    original = training.make_loader

    def capture(dataset, config, training=False, epoch=0):
        loaders.append((dataset, training, epoch))
        if epoch == 1:
            assert len((run / "metrics.jsonl").read_text().splitlines()) == 1
        return original(dataset, config, training=training, epoch=epoch)

    with patch("dual_payload.training.make_loader", side_effect=capture):
        training.train_main(["--config", str(path), "--overfit8"])
    return run, config, manifest_path, loaders


def test_overfit_cli_selection_and_shared_train_validation_pairs(overfit_run):
    run, config, manifest_path, loaders = overfit_run
    expected, selection = training.make_overfit8_dataset(manifest_path, config["seed"])
    assert json.loads((run / "overfit8_selection.json").read_text()) == selection
    assert len(loaders) == 4
    train, val = loaders[0][0], loaders[1][0]
    assert train is val and len(train) == 8
    assert all(dataset is train for dataset, _, _ in loaders)
    assert [training for _, training, _ in loaders] == [True, False, True, False]
    for i in range(8):
        assert torch.equal(train[i]["message"], val[i]["message"])
        assert torch.equal(train[i]["message"], expected[i]["message"])
        assert train[i]["path"] == val[i]["path"]


def test_overfit_two_epoch_history_keeps_all_actual_metrics(overfit_run):
    run, _, _, _ = overfit_run
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 2
    for epoch, record in enumerate(records):
        assert record["epoch"] == epoch and record["global_step"] == 4 * (epoch + 1)
        assert record["train_samples"] == 8 and record["mode"] == "overfit8"
        assert set(record["train_loss"]) == LOSS_KEYS
        assert set(record["validation"]) == METRIC_KEYS | {f"loss_{key}" for key in LOSS_KEYS}
        json.dumps(record, allow_nan=False)
    assert json.loads((run / "validation.json").read_text()) == records[-1]["validation"]


def test_overfit_checkpoint_flag_and_loading_compatibility(overfit_run):
    run, _, _, _ = overfit_run
    for name in ("last.pt", "best.pt", "best_message_ber.pt"):
        state = load_checkpoint(run / name)
        assert state["overfit8"] is True
        assert state["synthetic"] is False  # --overfit8 reads a prepared manifest, not --smoke.
        assert state["config"]["train"]["stage"] == "joint_256"
        assert state["schema"] == "medical-v3-256-v1"
    last = load_checkpoint(run / "last.pt")
    assert last["epoch"] == 1 and last["global_step"] == 8
    assert all(v["step"].item() == 8 for v in last["optimizer"]["state"].values())
    model, config, digest = model_from_file(run / "last.pt")
    assert model.message_bits == 256 and len(digest) == 32
    engine = ProtocolV1(run / "last.pt")
    assert engine.model.message_bits == 256 and engine.model_id == digest


def test_normal_train_still_random_through_cli_branch(tmp_path, monkeypatch):
    train = manifest(tmp_path, name="train")
    val = manifest(tmp_path, name="val")
    path, _ = config_file(tmp_path, train=train, val=val)

    class DataChecked(Exception):
        pass

    original = training.ManifestDataset
    built = []

    def inspect(*args, **kwargs):
        ds = original(*args, **kwargs)
        built.append(ds)
        return ds

    def stop_before_model(*args):
        assert built[0].training is True and built[1].training is False
        values = [built[0][0]["message"] for _ in range(4)]
        assert len({v.numpy().tobytes() for v in values}) == 4
        assert torch.equal(built[1][0]["message"], fixed_message(0, 12026))
        raise DataChecked

    monkeypatch.setattr(training, "ManifestDataset", inspect)
    monkeypatch.setattr(training, "DualPayloadSystem", stop_before_model)
    with pytest.raises(DataChecked):
        training.train_main(["--config", str(path)])


@pytest.mark.parametrize("overlap", ["path", "patient"])
def test_normal_isolation_not_relaxed(tmp_path, overlap):
    train = manifest(tmp_path, name="train")
    val = train if overlap == "path" else manifest(tmp_path, name="val")
    if overlap == "patient":
        rows = json.loads(val.read_text())
        rows["samples"][0]["patient_id"] = "train-p0"
        val.write_text(json.dumps(rows))
    path, _ = config_file(tmp_path, train=train, val=val)
    with patch("dual_payload.training.DualPayloadSystem", side_effect=AssertionError("must reject first")):
        with pytest.raises(ValueError, match="overlap"):
            training.train_main(["--config", str(path)])
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("mode", ["--smoke", "--overfit8"])
def test_output_directory_must_remain_new_or_empty(tmp_path, mode):
    train = manifest(tmp_path)
    path, _ = config_file(tmp_path, train=train)
    run = tmp_path / "run"
    run.mkdir()
    marker = run / "metrics.jsonl"
    marker.write_text('{"existing": true}\n')
    with pytest.raises(ValueError, match="new or empty"):
        training.train_main(["--config", str(path), mode])
    assert marker.read_text() == '{"existing": true}\n'
