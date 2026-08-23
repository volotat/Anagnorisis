"""
Tests for the model fingerprint in anagnorisis_core/embedder.py

This exists because of a real bug: the hash used to be computed from the loaded
``state_dict()``, and jina-v5-omni ships no trained audio LoRA — PEFT fills in
384 ``lora_A`` tensors randomly on every load. Those tensors are inert (their
``lora_B`` pairs are all zero), so embeddings were perfectly reproducible, but
the *hash* changed on every load. Every restart and every idle respawn minted
fresh cache keys, so the content, proxy and metadata caches were all rebuilt
from scratch each time, forever.

The rule these tests pin down: the fingerprint must depend on the weight files
and the encode settings, and on nothing else — in particular not on anything
that only exists after the model is loaded.
"""
import os
import pytest
from omegaconf import OmegaConf

from anagnorisis_core.models.embedder import _OmniEmbedderImpl


@pytest.fixture
def model_dir(tmp_path):
    """A stand-in for a downloaded model snapshot."""
    d = tmp_path / "jinaai__jina-embeddings-v5-omni-small"
    d.mkdir()
    (d / "config.json").write_text('{"hidden_size": 1024}')
    (d / "model.safetensors").write_bytes(b"\x01\x02\x03" * 1000)
    (d / "tokenizer.json").write_text('{"vocab": {}}')
    sub = d / "1_Pooling"
    sub.mkdir()
    (sub / "config.json").write_text('{"pooling_mode": "mean"}')
    return d


def make_impl(**overrides):
    cfg = OmegaConf.create({
        'embedder': {
            'model_name': 'jinaai/jina-embeddings-v5-omni-small',
            'task': 'retrieval',
            'embedding_dimension': 1024,
            **overrides,
        }
    })
    return _OmniEmbedderImpl(cfg)


class TestStability:
    def test_repeated_calls_agree(self, model_dir):
        impl = make_impl()
        hashes = {impl._calculate_model_hash(str(model_dir)) for _ in range(5)}
        assert len(hashes) == 1

    def test_independent_instances_agree(self, model_dir):
        """The restart case: a fresh worker must reproduce the same key."""
        a = make_impl()._calculate_model_hash(str(model_dir))
        b = make_impl()._calculate_model_hash(str(model_dir))
        assert a == b

    def test_does_not_require_a_loaded_model(self, model_dir):
        """The guard against the original bug.

        The fingerprint must be computable with ``self.model is None``. If this
        ever starts touching the loaded model again, the random audio LoRA
        leaks back into the cache key.
        """
        impl = make_impl()
        assert impl.model is None
        assert impl._calculate_model_hash(str(model_dir))

    def test_transient_download_files_are_ignored(self, model_dir):
        """A lock or half-written shard must not change the key."""
        before = make_impl()._calculate_model_hash(str(model_dir))
        (model_dir / "model.safetensors.lock").write_text("")
        (model_dir / "model.safetensors.incomplete").write_bytes(b"partial")
        (model_dir / ".DS_Store").write_bytes(b"junk")
        pycache = model_dir / "__pycache__"
        pycache.mkdir()
        (pycache / "mod.cpython-311.pyc").write_bytes(b"bytecode")
        assert make_impl()._calculate_model_hash(str(model_dir)) == before


class TestSensitivity:
    def test_changed_weights_change_the_hash(self, model_dir):
        before = make_impl()._calculate_model_hash(str(model_dir))
        (model_dir / "model.safetensors").write_bytes(b"\x09\x08\x07" * 1000)
        assert make_impl()._calculate_model_hash(str(model_dir)) != before

    def test_resized_weights_change_the_hash(self, model_dir):
        """Size is hashed, so a change beyond the sampled ends is still caught."""
        before = make_impl()._calculate_model_hash(str(model_dir))
        with open(model_dir / "model.safetensors", "ab") as fh:
            fh.write(b"\x00")
        assert make_impl()._calculate_model_hash(str(model_dir)) != before

    def test_added_file_changes_the_hash(self, model_dir):
        before = make_impl()._calculate_model_hash(str(model_dir))
        (model_dir / "model-00002-of-00002.safetensors").write_bytes(b"more")
        assert make_impl()._calculate_model_hash(str(model_dir)) != before

    def test_task_changes_the_hash(self, model_dir):
        """A different LoRA produces different vectors from identical files."""
        a = make_impl(task='retrieval')._calculate_model_hash(str(model_dir))
        b = make_impl(task='classification')._calculate_model_hash(str(model_dir))
        assert a != b

    def test_truncation_dim_changes_the_hash(self, model_dir):
        """Matryoshka truncation changes the vector's length, so also its key."""
        a = make_impl(embedding_dimension=1024)._calculate_model_hash(str(model_dir))
        b = make_impl(embedding_dimension=512)._calculate_model_hash(str(model_dir))
        assert a != b

    def test_model_name_changes_the_hash(self, model_dir):
        a = make_impl(model_name='jinaai/jina-embeddings-v5-omni-small')
        b = make_impl(model_name='some/other-model')
        assert a._calculate_model_hash(str(model_dir)) != b._calculate_model_hash(str(model_dir))


class TestRobustness:
    def test_missing_directory_still_yields_a_stable_hash(self, tmp_path):
        """If the snapshot cannot be read we must still key the cache
        consistently — degrade to the settings, never to a random value."""
        missing = str(tmp_path / "not-downloaded-yet")
        a = make_impl()._calculate_model_hash(missing)
        b = make_impl()._calculate_model_hash(missing)
        assert a == b and len(a) == 32


class TestHashWithoutLoading:
    """The proxy must be able to name the model without starting the worker.

    Cache keys contain the embedder's identity, and searching deliberately never
    loads the embedding model. So if the proxy can only report its hash after the
    worker has run, a fresh process builds different keys than the indexer did and
    finds nothing it previously indexed — which is exactly what happened.
    """

    def test_proxy_reports_a_hash_with_no_worker_started(self, model_dir, monkeypatch):
        from anagnorisis_core.models.embedder import OmniEmbedder, compute_model_hash

        cfg = OmegaConf.create({
            'main': {'embedding_models_path': str(model_dir.parent)},
            'embedder': {'model_name': 'jinaai/jina-embeddings-v5-omni-small',
                         'task': 'retrieval', 'embedding_dimension': 1024},
        })
        OmniEmbedder._instance = None
        emb = OmniEmbedder(cfg)

        # Starting a subprocess here would be the bug, not the fix.
        def refuse(*a, **k):
            raise AssertionError('reading model_hash must not start the worker')
        monkeypatch.setattr(emb, '_ensure_process_running', refuse)

        assert emb.model_hash == compute_model_hash(cfg, str(model_dir.parent))
        assert emb._process is None

    def test_the_worker_and_the_proxy_agree(self, model_dir):
        """Two implementations of this hash would be the original bug again."""
        from anagnorisis_core.models.embedder import compute_model_hash

        cfg = OmegaConf.create({
            'main': {'embedding_models_path': str(model_dir.parent)},
            'embedder': {'model_name': 'jinaai/jina-embeddings-v5-omni-small',
                         'task': 'retrieval', 'embedding_dimension': 1024},
        })
        impl = _OmniEmbedderImpl(cfg)
        assert impl._calculate_model_hash(str(model_dir)) == \
            compute_model_hash(cfg, str(model_dir.parent))
