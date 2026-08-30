"""
Tests for anagnorisis_core/sinks.py and the describe() flow.

No models: the descriptor and the embedding phase are stubbed, because what is
worth pinning down here is the *policy* — what gets written, what is left alone,
and what is deliberately absent from a published description.
"""
import os
import pytest
from omegaconf import OmegaConf

from anagnorisis_core import api
from anagnorisis_core.media.media_types import bundled_media_types_dir
from anagnorisis_core.storage.sinks import MetaSink, build_sink


@pytest.fixture
def cfg(tmp_path):
    mt = bundled_media_types_dir()
    conf = OmegaConf.load(os.path.join(mt, 'media_types.yaml'))
    return OmegaConf.merge(conf, OmegaConf.create({
        'main': {'cache_path': str(tmp_path / 'cache'),
                 'embedding_models_path': str(tmp_path / 'models'),
                 'media_types_path': mt},
        'embedder': {'model_name': 'jinaai/test-embedder', 'embedding_dimension': 1024},
        'omni': {'model_name': 'google/test-descriptor'},
    }))


@pytest.fixture
def media(tmp_path):
    """A small tree with a describable file and one to be ignored."""
    d = tmp_path / 'data'
    (d / 'sub').mkdir(parents=True)
    (d / 'notes.txt').write_text('A cat asleep on a mat.', encoding='utf-8')
    (d / 'sub' / 'more.txt').write_text('Another note.', encoding='utf-8')
    (d / 'ignore.xyz').write_text('unknown extension', encoding='utf-8')
    return d


@pytest.fixture
def media_needing_model(tmp_path):
    """A tree whose file is too long to keep verbatim.

    Short text is its own description and never reaches the descriptor, which
    is deliberate but makes it useless for testing anything about a model being
    loaded, released or failing to load.
    """
    d = tmp_path / 'data'
    d.mkdir(parents=True)
    (d / 'essay.txt').write_text('A cat asleep on a mat. ' * 400, encoding='utf-8')
    return d


@pytest.fixture
def stub(monkeypatch):
    """Stand in for both models; records which files were described."""
    described = []

    class FakeDescriptor:
        def __init__(self, cfg): pass
        def initiate(self, models_folder): pass
        def unload(self): pass
        def describe_text(self, text, prompt=None):
            described.append(text[:20])
            return 'A sleeping cat.'

    monkeypatch.setattr(api, 'OmniDescriptor', FakeDescriptor)
    monkeypatch.setattr(api, '_embed_batch', lambda *a, **k: None)
    monkeypatch.setattr(api, 'get_omni_embedder', lambda cfg: type('E', (), {'unload': lambda self: None})())
    return described


class TestCollectFiles:
    def test_finds_known_extensions_recursively(self, cfg, media):
        found = api.collect_files([str(media)], cfg=cfg)
        assert len(found) == 2, found
        assert not any(f.endswith('.xyz') for f in found), 'unknown types must be skipped'

    def test_no_recursive_stays_shallow(self, cfg, media):
        found = api.collect_files([str(media)], cfg=cfg, recursive=False)
        assert [os.path.basename(f) for f in found] == ['notes.txt']


class TestMetaSinkPolicy:
    def test_writes_a_sidecar(self, cfg, media, stub):
        report = api.describe([str(media)], cfg=cfg, sink=MetaSink(cfg))
        assert report.written == 2, report
        assert os.path.isfile(str(media / 'notes.txt.meta'))

    def test_second_run_writes_nothing(self, cfg, media, stub):
        api.describe([str(media)], cfg=cfg, sink=MetaSink(cfg))
        report = api.describe([str(media)], cfg=cfg, sink=MetaSink(cfg))
        assert report.written == 0 and report.already_done == 2, report

    def test_never_overwrites_a_handwritten_sidecar(self, cfg, media, stub):
        target = media / 'notes.txt.meta'
        target.write_text('My grandfather, 1963. Do not touch.', encoding='utf-8')
        api.describe([str(media)], cfg=cfg, sink=MetaSink(cfg))
        assert target.read_text(encoding='utf-8') == 'My grandfather, 1963. Do not touch.'

    def test_deleting_regenerates(self, cfg, media, stub):
        api.describe([str(media)], cfg=cfg, sink=MetaSink(cfg))
        os.unlink(str(media / 'notes.txt.meta'))
        report = api.describe([str(media)], cfg=cfg, sink=MetaSink(cfg))
        assert report.written == 1, 'deletion is how you ask for a fresh description'


class TestWhatAPublishedDescriptionContains:
    def test_has_the_description_and_a_stamp(self, cfg, media, stub):
        api.describe([str(media)], cfg=cfg, sink=MetaSink(cfg))
        text = (media / 'notes.txt.meta').read_text(encoding='utf-8')
        assert text.startswith('Anagnorisis-Generated:')
        assert text.split('\n')[0].count('\n') == 0, 'the stamp must be one line'
        # These fixtures are a sentence long, so the sidecar carries the words
        # themselves rather than a summary of them — see TestTheSizeRule.
        assert 'A cat asleep on a mat.' in text
        assert '# Internal metadata:' in text

    def test_does_not_publish_the_path(self, cfg, media, stub):
        api.describe([str(media)], cfg=cfg, sink=MetaSink(cfg))
        text = (media / 'notes.txt.meta').read_text(encoding='utf-8')
        assert 'File Path:' not in text, "a published description must not leak the layout"
        assert str(media) not in text

    def test_does_not_fold_itself_back_in(self, cfg, media, stub):
        """The sidecar must not contain a previous sidecar's contents."""
        api.describe([str(media)], cfg=cfg, sink=MetaSink(cfg))
        text = (media / 'notes.txt.meta').read_text(encoding='utf-8')
        assert '# External metadata from' not in text

    def test_no_partial_files_left_behind(self, cfg, media, stub):
        api.describe([str(media)], cfg=cfg, sink=MetaSink(cfg))
        leftovers = [p for p in os.listdir(media) if p.endswith('.partial')]
        assert leftovers == []


class TestStability:
    def test_a_changing_file_is_not_described(self, cfg, media, stub):
        """A half-copied file described once would be wrong forever."""
        state = {}
        report = api.describe([str(media)], cfg=cfg, sink=MetaSink(cfg),
                              require_stable=True, _stability_state=state)
        assert report.written == 0 and report.skipped_unstable == 2, report

        # Unchanged on the next pass, so now it is safe.
        report = api.describe([str(media)], cfg=cfg, sink=MetaSink(cfg),
                              require_stable=True, _stability_state=state)
        assert report.written == 2, report


class TestSinkFactory:
    def test_unknown_sink_is_rejected_by_name(self, cfg):
        with pytest.raises(ValueError, match="unknown sink"):
            build_sink('elsewhere', cfg=cfg)

    def test_cache_sink_requires_the_app_side_search(self, cfg):
        with pytest.raises(ValueError, match="MetadataSearch"):
            build_sink('cache', cfg=cfg)


class TestModelSwitching:
    """Each embed/describe cycle costs a load of both models, so the pipeline
    must not switch more than once per batch — and must not hold both at once."""

    def test_describe_does_not_call_initiate_on_the_embedder(self, cfg, media, monkeypatch):
        """`initiate()` loads the model and then unloads it again.

        Calling it before doing work costs a load, an unload and then a reload. It
        used to be called once per media type per batch.
        """
        calls = []

        class FakeEngine:
            def initiate(self, *a, **k):
                calls.append('initiate')
            def process_files(self, files, generate_embs_if_not_in_cache=True):
                calls.append('process_files')
                return [None for _ in files]

        class FakeDescriptor:
            def __init__(self, cfg): pass
            def initiate(self, models_folder): pass
            def unload(self): pass
            def describe_text(self, text, prompt=None): return 'A sleeping cat.'

        monkeypatch.setattr('anagnorisis_core.search.content_search.get_content_search',
                            lambda cfg, name: FakeEngine())
        # Stub the descriptor too: unstubbed it would try to download a model
        # named in the test config, which is both slow and a test touching the
        # network.
        monkeypatch.setattr(api, 'OmniDescriptor', FakeDescriptor)
        api.describe([str(media)], cfg=cfg, sink=MetaSink(cfg))
        assert 'initiate' not in calls, 'initiate() loads then unloads — pure waste here'

    def test_the_sidecar_is_written_after_the_descriptor_is_released(
            self, cfg, media_needing_model, monkeypatch):
        """Assembly can touch the caches, so it must not run with the descriptor
        resident — that is how both models ended up in memory together."""
        order = []

        class FakeDescriptor:
            def __init__(self, cfg): pass
            def initiate(self, models_folder): order.append('descriptor_loaded')
            def unload(self): order.append('descriptor_unloaded')
            def describe_text(self, text, prompt=None):
                order.append('described')
                return 'A sleeping cat.'

        class RecordingSink(MetaSink):
            def write(self, file_path, auto_description):
                order.append('written')
                return super().write(file_path, auto_description)

        monkeypatch.setattr(api, 'OmniDescriptor', FakeDescriptor)
        monkeypatch.setattr(api, '_embed_batch', lambda *a, **k: None)
        monkeypatch.setattr(api, 'get_omni_embedder',
                            lambda cfg: type('E', (), {'unload': lambda self: None})())

        api.describe([str(media_needing_model)], cfg=cfg, sink=RecordingSink(cfg))

        assert 'written' in order and 'descriptor_unloaded' in order, order
        assert order.index('descriptor_unloaded') < order.index('written'), (
            f'the sidecar was written while the descriptor was still loaded: {order}')

    def test_short_text_does_not_load_the_descriptor(self, cfg, media, monkeypatch):
        """Short text is its own description, so nothing should be loaded.

        The load costs about 13 seconds and the model would never be called:
        body_for_text returns the words unchanged below the verbatim limit.
        """
        loads = []

        class FakeDescriptor:
            def __init__(self, cfg): pass
            def initiate(self, models_folder): loads.append('load')
            def unload(self): pass
            def describe_text(self, text, prompt=None):
                raise AssertionError('the model was called for short text')

        monkeypatch.setattr(api, 'OmniDescriptor', FakeDescriptor)
        monkeypatch.setattr(api, '_embed_batch', lambda *a, **k: None)
        monkeypatch.setattr(api, 'get_omni_embedder',
                            lambda cfg: type('E', (), {'unload': lambda self: None})())

        report = api.describe([str(media)], cfg=cfg, sink=MetaSink(cfg))

        assert loads == [], f'the descriptor was loaded for short text: {loads}'
        assert report.written == 2, report

    def test_a_type_nothing_can_describe_is_skipped_not_failed(self, cfg, tmp_path,
                                                               monkeypatch):
        """A PDF has no describe method, so it is not a failure — nothing was
        ever attempted. It must also not drag the descriptor into memory."""
        d = tmp_path / 'docs'
        d.mkdir()
        (d / 'paper.pdf').write_bytes(b'%PDF-1.4 not really a pdf')

        loads = []

        class FakeDescriptor:
            def __init__(self, cfg): pass
            def initiate(self, models_folder): loads.append('load')
            def unload(self): pass

        monkeypatch.setattr(api, 'OmniDescriptor', FakeDescriptor)
        monkeypatch.setattr(api, '_embed_batch', lambda *a, **k: None)
        monkeypatch.setattr(api, 'get_omni_embedder',
                            lambda cfg: type('E', (), {'unload': lambda self: None})())

        report = api.describe([str(d)], cfg=cfg, sink=MetaSink(cfg))

        assert loads == [], f'loaded a model for a type it cannot describe: {loads}'
        assert report.failed == 0, report
        assert report.skipped_unsupported == 1, report

    def test_batch_size_zero_means_a_single_pass(self, cfg, media_needing_model, monkeypatch):
        """The fewest possible switches: two model loads for the whole run."""
        loads = []

        class FakeDescriptor:
            def __init__(self, cfg): pass
            def initiate(self, models_folder): loads.append('load')
            def unload(self): pass
            def describe_text(self, text, prompt=None): return 'A sleeping cat.'

        monkeypatch.setattr(api, 'OmniDescriptor', FakeDescriptor)
        monkeypatch.setattr(api, '_embed_batch', lambda *a, **k: None)
        monkeypatch.setattr(api, 'get_omni_embedder',
                            lambda cfg: type('E', (), {'unload': lambda self: None})())

        api.describe([str(media_needing_model)], cfg=cfg, sink=MetaSink(cfg), batch_size=0)
        assert len(loads) == 1, f'expected one descriptor load, got {len(loads)}'


class TestCompilationCacheDirs:
    """numba and torch both want a writable place for compiled artefacts, and a
    non-root container user has neither site-packages nor a home directory.

    numba *raises* (`no locator available`), so every audio file fails to be
    described with a traceback that points at librosa and never mentions
    permissions. Torch only warns and disables caching, so it costs time instead.
    """

    def test_both_variables_land_under_the_cache_path(self, cfg, monkeypatch):
        from anagnorisis_core.config import ensure_writable_cache_dirs
        for var in ('NUMBA_CACHE_DIR', 'PYTORCH_KERNEL_CACHE_PATH'):
            monkeypatch.delenv(var, raising=False)

        ensure_writable_cache_dirs(cfg)

        for var in ('NUMBA_CACHE_DIR', 'PYTORCH_KERNEL_CACHE_PATH'):
            target = os.environ.get(var)
            assert target, f'{var} was not set'
            assert target.startswith(str(cfg.main.cache_path)), (var, target)
            assert os.path.isdir(target), target

    def test_an_explicit_setting_is_respected(self, cfg, monkeypatch, tmp_path):
        """A deployment that places these itself must win."""
        from anagnorisis_core.config import ensure_writable_cache_dirs
        chosen = str(tmp_path / 'mine')
        monkeypatch.setenv('NUMBA_CACHE_DIR', chosen)
        monkeypatch.delenv('PYTORCH_KERNEL_CACHE_PATH', raising=False)

        ensure_writable_cache_dirs(cfg)

        assert os.environ['NUMBA_CACHE_DIR'] == chosen, 'must not override the user'
        assert os.environ['PYTORCH_KERNEL_CACHE_PATH'].startswith(
            str(cfg.main.cache_path)), 'the other one should still be set'

    def test_both_workers_call_it_first(self):
        """The call has to come before the worker does anything else: librosa is
        imported lazily further down and reads the environment as it finds it.

        Checked on the parsed statements rather than the source text — the first
        version of this test matched the word "librosa" in the comment above the
        call and failed on correct code.
        """
        import ast
        import inspect
        import textwrap

        from anagnorisis_core.models import descriptor, embedder

        def calls_ensure(node):
            return any(isinstance(n, ast.Call)
                       and getattr(n.func, 'id', None) == 'ensure_writable_cache_dirs'
                       for n in ast.walk(node))

        for module in (descriptor, embedder):
            tree = ast.parse(textwrap.dedent(inspect.getsource(module._worker_loop)))
            body = tree.body[0].body
            positions = [i for i, stmt in enumerate(body) if calls_ensure(stmt)]
            assert positions, f'{module.__name__} never fixes the cache dirs'
            # Only setproctitle and the import of the helper may precede it.
            allowed = {'setproctitle', 'ensure_writable_cache_dirs'}
            for stmt in body[:positions[0]]:
                names = {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name)}
                names |= {a.name.split('.')[0] for n in ast.walk(stmt)
                          if isinstance(n, ast.Import) for a in n.names}
                assert names <= allowed | {'cfg'}, (
                    f'{module.__name__} does {names} before fixing the cache dirs')


class TestWorkerCleanup:
    """A worker left running is not a leak of memory only — it stops the process
    from exiting at all, so a CLI run appears to finish and then hangs forever.
    Both halves of that are pinned here.
    """

    def test_a_failed_initiate_still_unloads_the_descriptor(
            self, cfg, media_needing_model, monkeypatch):
        """initiate() spawns the worker *before* it can fail — the VRAM guard
        raises from inside it — so the failure path has to unload. It did not,
        and a guarded run left a worker holding VRAM and hung on exit.
        """
        calls = []

        class FailingDescriptor:
            def __init__(self, cfg): pass
            def initiate(self, models_folder):
                calls.append('initiate')
                raise RuntimeError('only 736 MB of VRAM free, need 7168 MB')
            def unload(self): calls.append('unload')

        monkeypatch.setattr(api, 'OmniDescriptor', FailingDescriptor)
        monkeypatch.setattr(api, '_embed_batch', lambda *a, **k: None)
        monkeypatch.setattr(api, 'get_omni_embedder',
                            lambda cfg: type('E', (), {'unload': lambda self: None})())

        report = api.describe([str(media_needing_model)], cfg=cfg, sink=MetaSink(cfg))

        assert 'unload' in calls, (
            f'a failed initiate left the worker running: {calls}')
        assert report.written == 0 and report.failed > 0
        assert any('descriptor unavailable' in e for e in report.errors)

    def test_every_worker_process_is_a_daemon(self):
        """multiprocessing joins non-daemon children at interpreter exit, so any
        worker that outlives its owner blocks the process. Checked at the source
        so a newly added worker cannot reintroduce the hang.
        """
        import ast
        import pathlib

        # rglob, not glob: the workers live in anagnorisis_core/models/ now, and
        # a scan of the top level alone would find nothing and quietly pass on
        # the "no worker processes" branch.
        core = pathlib.Path(api.__file__).parent
        skip = {'tests', 'benchmarks'}
        found = []
        for path in sorted(p for p in core.rglob('*.py')
                           if not skip & set(p.relative_to(core).parts)):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not (isinstance(node.func, ast.Attribute)
                        and node.func.attr == 'Process'):
                    continue
                daemon = [kw for kw in node.keywords if kw.arg == 'daemon']
                ok = bool(daemon) and getattr(daemon[0].value, 'value', None) is True
                found.append((path.name, node.lineno, ok))

        assert found, 'no worker processes found — has the search broken?'
        offenders = [f'{n}:{l}' for n, l, ok in found if not ok]
        assert not offenders, f'worker processes missing daemon=True: {offenders}'
