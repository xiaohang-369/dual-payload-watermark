from copy import deepcopy
import json

import pytest
import torch
from PIL import Image

from dual_payload.config import DEFAULT_CONFIG
from dual_payload.diagnostics import (diagnostic_main, evaluate_bank, evaluate_random_messages,
                                      fixed_codebook_banks, gradient_probe, inspect_image,
                                      message_banks, message_metrics, overfit, prior_baseline,
                                      random_message_bank, random_message_fit)
from dual_payload.transforms import rgb_to_ycbcr
from dual_payload.system import DualPayloadSystem
from dual_payload.training import load_checkpoint


def setup_model():
    config = deepcopy(DEFAULT_CONFIG)
    config["model"].update(channels=4, blocks=1)
    config["data"]["image_size"] = 16
    model = DualPayloadSystem(config["model"], config["channel"]).eval()
    return model, config


def test_message_banks_are_reproducible_disjoint_and_not_one_label_per_image():
    fit, heldout = message_banks(4, 31)
    assert fit.shape == heldout.shape == (4, 64)
    assert len({tuple(row.tolist()) for row in torch.cat((fit, heldout))}) == 8
    assert torch.equal(fit, message_banks(4, 31)[0])
    with pytest.raises(ValueError, match="at least two"):
        message_banks(1, 31)


def test_fixed_codebook_uses_independent_random_validation_bank():
    fit, validation = fixed_codebook_banks(20, 256, 31, 32)
    assert fit.shape == (20, 64)
    assert validation.shape == (256, 64)
    assert torch.equal(validation, random_message_bank(256, 32))
    assert not ({tuple(row.tolist()) for row in fit}
                & {tuple(row.tolist()) for row in validation})
    with pytest.raises(ValueError, match="at least two"):
        fixed_codebook_banks(1, 2, 31, 32)


def test_prior_baseline_exposes_constant_prediction_shortcut():
    fit = torch.zeros(4, 64)
    fit[0] = 1
    result = prior_baseline(fit, 1 - fit)
    assert result["fit_bank"]["ber"] == 0.25
    assert result["heldout_messages"]["ber"] == 0.75
    assert result["fit_bit_one_counts"] == [1] * 64
    assert result["fit_bit_one_frequency"] == [0.25] * 64


def test_message_metrics_can_report_exact_per_bit_errors():
    logits = torch.full((2, 64), -1.0)
    messages = torch.zeros(2, 64)
    messages[:, 0] = 1
    messages[0, 7] = 1
    result = message_metrics(logits, messages, include_per_bit=True)
    assert result["ber"] == pytest.approx(3 / 128)
    assert result["bit_errors"] == 3
    assert result["samples"] == 2
    assert result["bits"] == 128
    assert len(result["per_bit_ber"]) == len(result["per_bit_error_counts"]) == 64
    assert result["per_bit_ber"][0] == 1
    assert result["per_bit_ber"][7] == 0.5
    assert sum(result["per_bit_ber"]) / 64 == pytest.approx(result["ber"])
    json.dumps(result, allow_nan=False)


def test_random_message_evaluation_is_batching_invariant_and_has_controls():
    model, _ = setup_model()
    rgb = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        host = model.color_encoder(*rgb_to_ycbcr(rgb))["carrier"]
    validation = random_message_bank(5, 40)
    first = evaluate_random_messages(model, host, validation, torch.device("cpu"), 2)
    second = evaluate_random_messages(model, host, validation, torch.device("cpu"), 3)
    assert set(first) == {"matched", "mismatched_labels", "watermark_removed"}
    for key in first:
        assert first[key]["ber"] == pytest.approx(second[key]["ber"])
        assert first[key]["bce"] == pytest.approx(second[key]["bce"], abs=1e-7)
    assert first["matched"]["per_bit_ber"] == second["matched"]["per_bit_ber"]


def test_fixed_bank_evaluation_is_batching_invariant_and_has_controls():
    model, _ = setup_model()
    rgb = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        host = model.color_encoder(*rgb_to_ycbcr(rgb))["carrier"]
    messages = random_message_bank(5, 41)
    first = evaluate_bank(model, [host], messages, torch.device("cpu"), 2)
    second = evaluate_bank(model, [host], messages, torch.device("cpu"), 3)
    assert set(first) == {"matched", "mismatched_labels", "watermark_removed"}
    for key in first:
        assert first[key]["ber"] == pytest.approx(second[key]["ber"])
        assert first[key]["bce"] == pytest.approx(second[key]["bce"], abs=1e-7)
    assert first["matched"]["per_bit_ber"] == second["matched"]["per_bit_ber"]


def test_inspect_zero_head_has_no_message_carrier_or_logit_variation():
    model, _ = setup_model()
    fit, _ = message_banks(4, 32)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = inspect_image(model, torch.rand(1, 3, 16, 16), fit)
    assert result["embedding_message_variation_rms"] > 0
    for key in ("candidate_message_variation_rms", "projected_message_variation_rms",
                "residual_message_variation_rms", "logits_message_variation_rms",
                "single_flip_residual_diff_rms", "single_flip_logits_diff_rms"):
        assert result[key] == 0
    assert result["matched"] == result["watermark_removed"]
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    assert all(parameter.grad is None for parameter in model.parameters())
    json.dumps(result, allow_nan=False)


def test_gradient_probe_observes_actual_path_without_updating_weights():
    model, config = setup_model()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    fit, _ = message_banks(2, 33)
    report = gradient_probe(model, torch.rand(2, 3, 16, 16), fit, config)
    # A zero encoder head blocks the message MLP on the first backward, not the head itself.
    assert report["message"]["groups"]["message_branch"]["grad_l2"] == 0
    assert report["message"]["groups"]["watermark_encoder_head"]["grad_l2"] > 0
    assert report["message"]["activation_gradient_rms"]["candidate"] > 0
    assert report["total"]["all_parameters"]["finite"]
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    assert all(parameter.grad is None for parameter in model.parameters())
    assert len(model.watermark_encoder._forward_hooks) == 0
    json.dumps(report, allow_nan=False)


def test_nonzero_head_exposes_message_gradient_and_sensitivity():
    model, config = setup_model()
    with torch.no_grad():
        model.watermark_encoder.head.weight.normal_(0, 0.01)
    fit, _ = message_banks(2, 34)
    rgb = torch.rand(1, 3, 16, 16)
    assert inspect_image(model, rgb, fit)["residual_message_variation_rms"] > 0
    report = gradient_probe(model, rgb, fit[:1], config)
    assert report["message"]["groups"]["message_branch"]["grad_l2"] > 0


def test_overfit_only_changes_watermark_weights_and_uses_separate_format(tmp_path):
    model, config = setup_model()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    fit, heldout = message_banks(2, 35)
    history = overfit(model, [torch.rand(1, 3, 16, 16) for _ in range(2)], fit, heldout,
                      config, steps=2, batch_size=2, log_every=2,
                      device=torch.device("cpu"), run_dir=tmp_path)
    assert [event["diagnostic_step"] for event in history] == [0, 1, 2]
    for key, value in model.state_dict().items():
        if key.startswith(("color_encoder.", "color_decoder.")):
            assert torch.equal(value, before[key])
    assert not torch.equal(model.watermark_encoder.head.weight, before["watermark_encoder.head.weight"])
    assert "heldout_messages_same_images" in history[-1]
    assert history[-1]["training_presentations"] == 4
    assert history[-1]["pair_exposure_min"] == history[-1]["pair_exposure_max"] == 1
    assert history[-1]["pair_exposure_gap"] == 0
    assert history[-1]["pair_exposure_counts"] == [1, 1, 1, 1]
    assert set(history[-1]["fit_controls"]) == {"mismatched_labels", "watermark_removed"}
    assert len(history[-1]["heldout_messages_same_images"]["per_bit_ber"]) == 64
    with pytest.raises(ValueError, match="Unsupported checkpoint"):
        load_checkpoint(tmp_path / "diagnostic_weights.pt")


def test_random_message_fit_uses_unique_unseen_messages_and_preserves_color(tmp_path):
    model, config = setup_model()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    validation = random_message_bank(4, 36)
    captured = []

    def capture_training_messages(module, inputs):
        if module.training and torch.is_grad_enabled():
            captured.append(inputs[1].detach().cpu().clone())

    handle = model.watermark_encoder.register_forward_pre_hook(capture_training_messages)
    try:
        history = random_message_fit(
            model, torch.rand(1, 3, 16, 16), validation, config,
            steps=2, batch_size=2, eval_batch_size=2, log_every=2,
            device=torch.device("cpu"), run_dir=tmp_path, training_seed=37)
    finally:
        handle.remove()
    assert [event["diagnostic_step"] for event in history] == [0, 1, 2]
    assert [event["training_samples_seen"] for event in history] == [0, 2, 4]
    training = torch.cat(captured)
    assert len(training) == 4
    assert len({tuple(row.tolist()) for row in training}) == 4
    validation_keys = {tuple(row.tolist()) for row in validation}
    assert not validation_keys.intersection(tuple(row.tolist()) for row in training)
    metrics = history[-1]["validation_messages_same_image"]
    assert len(metrics["per_bit_ber"]) == 64
    assert sum(metrics["per_bit_ber"]) / 64 == pytest.approx(metrics["ber"])
    assert "mismatched_labels_same_image" in history[-1]
    assert "watermark_removed_same_image" in history[-1]
    for key, value in model.state_dict().items():
        if key.startswith(("color_encoder.", "color_decoder.")):
            assert torch.equal(value, before[key])
    assert not torch.equal(model.watermark_encoder.head.weight,
                           before["watermark_encoder.head.weight"])
    diagnostic = torch.load(tmp_path / "diagnostic_weights.pt", map_location="cpu",
                            weights_only=True)
    assert diagnostic["kind"] == "watermark_only_bce_random_messages"
    assert diagnostic["training_messages_seen"] == 4
    assert torch.equal(diagnostic["validation_messages"], validation)
    with pytest.raises(ValueError, match="Unsupported checkpoint"):
        load_checkpoint(tmp_path / "diagnostic_weights.pt")


def test_random_message_training_is_independent_of_global_rng_and_eval_frequency(tmp_path):
    first, config = setup_model()
    second, _ = setup_model()
    second.load_state_dict(first.state_dict())
    rgb = torch.rand(1, 3, 16, 16)
    validation = random_message_bank(3, 38)
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    random_message_fit(first, rgb, validation, config, steps=4, batch_size=2,
                       eval_batch_size=2, log_every=1, device=torch.device("cpu"),
                       run_dir=first_dir, training_seed=39)
    torch.manual_seed(999999)
    random_message_fit(second, rgb, validation, config, steps=4, batch_size=2,
                       eval_batch_size=2, log_every=4, device=torch.device("cpu"),
                       run_dir=second_dir, training_seed=39)
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])


def test_inspect_cli_preserves_source_and_rejects_existing_output(tmp_path):
    model, config = setup_model()
    source = tmp_path / "last.pt"
    torch.save({"format_version": 1, "model": model.state_dict(),
                "config": config, "global_step": 1000}, source)
    original = source.read_bytes()
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (16, 16), (100, 140, 120)).save(images / "one.png")
    output = tmp_path / "inspect"
    args = ["--checkpoint", str(source), "--data-dir", str(images), "--images", "1",
            "--messages", "2", "--device", "cpu", "--output-dir", str(output)]
    diagnostic_main(args)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["metadata"]["objective"] == "no_updates"
    assert source.read_bytes() == original
    assert not list(output.glob("*.pt"))
    with pytest.raises(ValueError, match="NEW directory"):
        diagnostic_main(args)
    for extra in (["--steps", "1"], ["--mode", "overfit"]):
        with pytest.raises(SystemExit):
            diagnostic_main(args + extra)


def test_random_messages_cli_validates_mode_and_preserves_source(tmp_path):
    model, config = setup_model()
    source = tmp_path / "last.pt"
    torch.save({"format_version": 1, "model": model.state_dict(),
                "config": config, "global_step": 1000}, source)
    original = source.read_bytes()
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (16, 16), (100, 140, 120)).save(images / "one.png")
    common = ["--checkpoint", str(source), "--data-dir", str(images),
              "--mode", "random_messages", "--device", "cpu"]
    with pytest.raises(SystemExit):
        diagnostic_main(common + ["--images", "1"])
    with pytest.raises(SystemExit):
        diagnostic_main(common + ["--images", "2", "--steps", "1"])
    with pytest.raises(SystemExit):
        diagnostic_main(common + ["--images", "1", "--steps", "1", "--messages", "2"])
    with pytest.raises(SystemExit):
        diagnostic_main(common + ["--images", "1", "--steps", "1",
                                  "--validation-messages", "1"])
    output = tmp_path / "random"
    args = common + ["--images", "1", "--steps", "1", "--validation-messages", "2",
                     "--output-dir", str(output)]
    diagnostic_main(args)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    metadata = report["metadata"]
    assert metadata["objective"] == "watermark_BCE_only_fresh_random_messages"
    assert metadata["host_policy"] == "single_cached_color_carrier"
    assert metadata["training_message_policy"] == "fresh_unique_random_each_step_excluding_validation"
    assert metadata["validation_message_count"] == 2
    assert "fit_messages" not in metadata
    assert len(report["evaluations"][-1]["validation_messages_same_image"]["per_bit_ber"]) == 64
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match="NEW directory"):
        diagnostic_main(args)


def test_overfit_cli_supports_large_independent_validation_bank(tmp_path):
    model, config = setup_model()
    source = tmp_path / "last.pt"
    torch.save({"format_version": 1, "model": model.state_dict(),
                "config": config, "global_step": 1000}, source)
    original = source.read_bytes()
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (16, 16), (100, 140, 120)).save(images / "one.png")
    output = tmp_path / "fixed_codebook"
    args = ["--checkpoint", str(source), "--data-dir", str(images),
            "--mode", "overfit", "--device", "cpu", "--images", "1",
            "--messages", "3", "--validation-messages", "5",
            "--eval-batch-size", "2", "--steps", "1", "--output-dir", str(output)]
    diagnostic_main(args)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    metadata = report["metadata"]
    assert metadata["fit_message_count"] == 3
    assert metadata["validation_message_count"] == 5
    assert metadata["validation_message_policy"] == (
        "fixed_independent_bank_shared_with_random_messages")
    expected = random_message_bank(5, config["seed"] + 800000).int().tolist()
    assert metadata["heldout_messages"] == expected
    final = report["evaluations"][-1]
    assert final["training_presentations"] == 2
    assert final["pair_exposure_min"] == 0
    assert final["pair_exposure_max"] == 1
    assert set(final["heldout_controls"]) == {"mismatched_labels", "watermark_removed"}
    diagnostic = torch.load(output / "diagnostic_weights.pt", map_location="cpu",
                            weights_only=True)
    assert sorted(diagnostic["pair_exposure_counts"].tolist()) == [0, 1, 1]
    assert source.read_bytes() == original

    legacy_output = tmp_path / "legacy_overfit"
    legacy_args = ["--checkpoint", str(source), "--data-dir", str(images),
                   "--mode", "overfit", "--device", "cpu", "--images", "1",
                   "--messages", "3", "--steps", "1", "--output-dir", str(legacy_output)]
    diagnostic_main(legacy_args)
    legacy = json.loads((legacy_output / "report.json").read_text(encoding="utf-8"))
    expected_fit, expected_heldout = message_banks(3, config["seed"] + 700000)
    assert legacy["metadata"]["fit_messages"] == expected_fit.int().tolist()
    assert legacy["metadata"]["heldout_messages"] == expected_heldout.int().tolist()
    assert legacy["metadata"]["validation_message_policy"] == (
        "legacy_disjoint_suffix_from_fit_rng")
    assert source.read_bytes() == original
