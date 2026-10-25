"""
Tests for how the descriptor picks its precision and its free-VRAM threshold.

This exists because one public config has to serve very different cards. The
project was tuned on an 8 GB NVIDIA card, where only 4-bit NF4 fits. On a 16 GB
AMD card the same settings were actively harmful: bitsandbytes had no kernel
that ran there, and instead of raising it called exit(1), so the descriptor
worker died on every load and that machine never wrote a single description.
The unquantized model, which fits that card comfortably, needs no bitsandbytes
at all.

So 'auto' must choose by the card, never ask bitsandbytes when it does not need
to, and hash exactly like the explicit setting it resolves to — otherwise moving
an existing install to 'auto' would throw away its cached descriptions.
"""
import subprocess

import pytest
from omegaconf import OmegaConf

from anagnorisis_core.models.descriptor import (
    VRAM_NEEDED_4BIT_MB,
    VRAM_NEEDED_4BIT_OFFLOADED_MB,
    VRAM_RESERVE_MB,
    VRAM_WORKING_MB,
    _OmniDescriptorImpl,
    checkpoint_size_mb,
    choose_precision,
    four_bit_quantization_works,
)

GEMMA_WEIGHTS_MB = 9772    # google/gemma-4-E2B-it, bf16 safetensors
LAPTOP_8GB_MB = 7967       # what an "8 GB" RTX 3070 Laptop reports
CARD_12GB_MB = 12288
CARD_16GB_MB = 16368


def never_asked():
    raise AssertionError('bitsandbytes must not be probed for this choice')


class TestAuto:
    def test_8gb_card_gets_4bit_with_the_measured_threshold(self):
        choice = choose_precision('auto', 'auto', LAPTOP_8GB_MB, GEMMA_WEIGHTS_MB)
        assert choice.load_in_4bit is True
        assert choice.min_free_vram_mb == VRAM_NEEDED_4BIT_MB
        assert choice.offload_to_ram is False

    def test_16gb_card_runs_unquantized_without_touching_bitsandbytes(self):
        choice = choose_precision('auto', 'auto', CARD_16GB_MB, GEMMA_WEIGHTS_MB,
                                  four_bit_works=never_asked)
        assert choice.load_in_4bit is False
        assert choice.min_free_vram_mb == GEMMA_WEIGHTS_MB + VRAM_WORKING_MB
        assert choice.offload_to_ram is False

    def test_12gb_card_is_too_tight_for_unquantized(self):
        # It would hold the model, but with less than the reserve left for a
        # desktop or anything else sharing the card.
        assert GEMMA_WEIGHTS_MB + VRAM_WORKING_MB < CARD_12GB_MB
        choice = choose_precision('auto', 'auto', CARD_12GB_MB, GEMMA_WEIGHTS_MB)
        assert choice.load_in_4bit is True

    def test_broken_bitsandbytes_falls_back_to_unquantized_with_offload(self):
        choice = choose_precision('auto', 'auto', CARD_12GB_MB, GEMMA_WEIGHTS_MB,
                                  four_bit_works=lambda: False)
        assert choice.load_in_4bit is False
        assert choice.offload_to_ram is True
        # Waits for a near-empty card instead of for room it will never have.
        assert choice.min_free_vram_mb == CARD_12GB_MB - VRAM_RESERVE_MB

    def test_no_gpu_means_unquantized_and_no_vram_check(self):
        choice = choose_precision('auto', 'auto', None, GEMMA_WEIGHTS_MB,
                                  four_bit_works=never_asked)
        assert choice.load_in_4bit is False
        assert choice.min_free_vram_mb is None

    def test_offloaded_embeddings_lower_the_4bit_threshold(self):
        choice = choose_precision('auto', 'auto', LAPTOP_8GB_MB, GEMMA_WEIGHTS_MB,
                                  cpu_offload_embeddings=True)
        assert choice.min_free_vram_mb == VRAM_NEEDED_4BIT_OFFLOADED_MB

    @pytest.mark.parametrize('value', ['auto', 'AUTO', ' Auto '])
    def test_auto_is_case_and_space_insensitive(self, value):
        assert choose_precision(value, value, CARD_16GB_MB, GEMMA_WEIGHTS_MB).load_in_4bit is False


class TestExplicitSettingsWin:
    def test_explicit_4bit_on_a_big_card(self):
        choice = choose_precision(True, 'auto', CARD_16GB_MB, GEMMA_WEIGHTS_MB,
                                  four_bit_works=never_asked)
        assert choice.load_in_4bit is True
        assert choice.min_free_vram_mb == VRAM_NEEDED_4BIT_MB

    def test_explicit_unquantized_on_a_small_card_offloads(self):
        choice = choose_precision(False, 'auto', LAPTOP_8GB_MB, GEMMA_WEIGHTS_MB,
                                  four_bit_works=never_asked)
        assert choice.load_in_4bit is False
        assert choice.offload_to_ram is True
        assert choice.min_free_vram_mb == LAPTOP_8GB_MB - VRAM_RESERVE_MB

    def test_explicit_threshold(self):
        assert choose_precision('auto', 5000, CARD_16GB_MB,
                                GEMMA_WEIGHTS_MB).min_free_vram_mb == 5000

    @pytest.mark.parametrize('value', [None, 0])
    def test_threshold_can_be_switched_off(self, value):
        assert choose_precision('auto', value, CARD_16GB_MB,
                                GEMMA_WEIGHTS_MB).min_free_vram_mb is None


@pytest.fixture
def model_dir(tmp_path):
    """A stand-in for a downloaded descriptor snapshot."""
    d = tmp_path / "google__gemma-4-E2B-it"
    d.mkdir()
    (d / "config.json").write_text('{"model_type": "gemma4"}')
    (d / "model.safetensors").write_bytes(b"\x00" * (3 * 2**20))
    (d / "tokenizer.json").write_text('{"vocab": {}}')
    return d


def test_checkpoint_size_counts_only_weights(model_dir):
    assert checkpoint_size_mb(str(model_dir)) == 3


class TestHashFollowsTheResolvedChoice:
    def _hash(self, model_dir, load_in_4bit, resolved_4bit):
        cfg = OmegaConf.create({'omni': {'model_name': 'google/gemma-4-E2B-it',
                                         'load_in_4bit': load_in_4bit}})
        impl = _OmniDescriptorImpl(cfg)
        impl.precision = choose_precision(load_in_4bit, 'auto', LAPTOP_8GB_MB, 9772,
                                          four_bit_works=lambda: resolved_4bit)
        return impl._calculate_model_hash(str(model_dir))

    def test_auto_landing_on_4bit_keeps_existing_descriptions(self, model_dir):
        assert self._hash(model_dir, 'auto', True) == self._hash(model_dir, True, True)

    def test_unquantized_descriptions_get_their_own_key(self, model_dir):
        assert self._hash(model_dir, 'auto', False) != self._hash(model_dir, True, True)


def test_a_bitsandbytes_crash_is_reported_not_raised(monkeypatch, capsys):
    """The real failure: an error line, then the libdrm noise printed after it."""
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout='', stderr=(
            "Error no kernel image is available for execution on the device "
            "at line 74 in file /src/csrc/ops.cu\n"
            "amdgpu.ids: No such file or directory\n"))
    monkeypatch.setattr(subprocess, 'run', fake_run)

    assert four_bit_quantization_works() is False
    assert 'no kernel image is available' in capsys.readouterr().out
