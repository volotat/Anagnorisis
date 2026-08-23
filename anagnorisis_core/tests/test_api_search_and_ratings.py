"""
Tests for the search, rating and ranking functions in anagnorisis_core/api.py.

These are the behaviours the README promises, so they are worth pinning down:
name search needs no index, unindexed files are dropped rather than ranked last,
your own ratings outrank the model's guesses, and a memory file survives the file
being renamed.

No models: name search and the rating paths need none, and the embedding-backed
modes are checked through their contract rather than by running a network.
"""
import os
import pytest
from omegaconf import OmegaConf

from anagnorisis_core import api
from anagnorisis_core.ratings import memory
from anagnorisis_core.media.media_types import bundled_media_types_dir
from anagnorisis_core.storage.soft_hash import get_file_soft_hash


@pytest.fixture
def cfg(tmp_path):
    mt = bundled_media_types_dir()
    conf = OmegaConf.load(os.path.join(mt, 'media_types.yaml'))
    return OmegaConf.merge(conf, OmegaConf.create({
        'main': {'cache_path': str(tmp_path / 'cache'),
                 'embedding_models_path': str(tmp_path / 'models'),
                 'personal_models_path': str(tmp_path / 'personal'),
                 'media_types_path': mt},
        'embedder': {'model_name': 'test/embedder', 'embedding_dimension': 1024},
        'omni': {'model_name': 'test/descriptor'},
    }))


@pytest.fixture
def media(tmp_path):
    d = tmp_path / 'media'
    (d / 'music').mkdir(parents=True)
    (d / 'notes about music.txt').write_text('songs I like', encoding='utf-8')
    (d / 'music' / 'shopping list.txt').write_text('milk, bread', encoding='utf-8')
    return d


class TestNameSearch:
    def test_needs_no_index_and_finds_by_filename(self, cfg, media):
        hits = api.search('music', [str(media)], cfg=cfg, mode='name')
        assert hits, 'name search must work with nothing indexed'
        assert os.path.basename(hits[0][0]) == 'notes about music.txt'

    def test_filename_hit_outranks_a_folder_hit(self, cfg, media):
        # Files are addressed as VFS URLs throughout, so results are keyed that way.
        hits = dict(api.search('music', [str(media)], cfg=cfg, mode='name'))
        by_name = hits[f"osfs://{media / 'notes about music.txt'}"]
        by_folder = hits[f"osfs://{media / 'music' / 'shopping list.txt'}"]
        assert by_name > by_folder, 'a name match should beat a path match'

    def test_an_unrelated_query_scores_low(self, cfg, media):
        hits = api.search('zzzzz', [str(media)], cfg=cfg, mode='name')
        assert all(score < 0.6 for _, score in hits), hits

    def test_limit_is_honoured(self, cfg, media):
        assert len(api.search('txt', [str(media)], cfg=cfg, mode='name', limit=1)) == 1

    def test_results_are_full_vfs_urls(self, cfg, media):
        """One spelling for a file, wherever it came from."""
        found = api.collect_files([str(media)], cfg=cfg)
        assert found, 'expected to find the media files'
        assert all(f.startswith('osfs:///') for f in found), found

    def test_hidden_directories_are_skipped(self, cfg, media):
        """Version control, thumbnail caches and trash are not user content —
        and with the meta sink, indexing them would write .meta files into them."""
        hidden = media / '.git'
        hidden.mkdir()
        (hidden / 'sneaky.txt').write_text('object data', encoding='utf-8')
        (media / '.thumbnails').mkdir()
        (media / '.thumbnails' / 'thumb.txt').write_text('cache', encoding='utf-8')

        found = api.collect_files([str(media)], cfg=cfg)
        assert not [f for f in found if '/.git/' in f or '/.thumbnails/' in f], found

    def test_a_remote_path_is_refused_not_downloaded(self, cfg):
        """Listing files must never be what triggers a download."""
        with pytest.raises(ValueError, match='remote server'):
            api.collect_files(['webdav://192.0.2.1:6001/share'], cfg=cfg)

    def test_an_osfs_url_is_local_and_accepted(self, cfg, media):
        by_url = api.collect_files([f'osfs://{media}'], cfg=cfg)
        by_path = api.collect_files([str(media)], cfg=cfg)
        assert by_url and by_url == by_path, 'both spellings must agree'

    def test_unknown_mode_is_rejected_by_name(self, cfg, media):
        with pytest.raises(ValueError, match='unknown search mode'):
            api.search('x', [str(media)], cfg=cfg, mode='telepathy')


class TestUnindexedFilesAreDropped:
    def test_metadata_search_returns_nothing_when_nothing_is_indexed(
            self, cfg, media, monkeypatch):
        """An unindexed file is unknown, not irrelevant — it must not rank last."""
        import anagnorisis_core.api as api_mod

        class FakeSearch:
            embedder = None
            def process_query(self, q):
                return [0.0]
            def process_files(self, files, generate_embs_if_not_in_cache=False):
                assert generate_embs_if_not_in_cache is False, \
                    'searching must never trigger embedding'
                return [None for _ in files]
            def compare(self, embeddings, query):
                import math
                return [math.nan for _ in embeddings]

        monkeypatch.setattr(
            'anagnorisis_core.search.metadata_search.get_metadata_search',
            lambda cfg: FakeSearch())
        assert api.search('anything', [str(media)], cfg=cfg, mode='metadata') == []


class TestRatingAndMemory:
    def test_rating_writes_a_memory_file_named_by_content(self, cfg, media):
        target = str(media / 'notes about music.txt')
        written = api.rate(target, 8.5, cfg=cfg, memory_dir=str(media / '..' / 'mem'))
        assert os.path.isfile(written)
        assert os.path.basename(written) == f'{get_file_soft_hash(target)}.md'

    def test_the_rating_is_line_one_and_nowhere_else(self, cfg, media, tmp_path):
        target = str(media / 'notes about music.txt')
        written = api.rate(target, 7.0, cfg=cfg, memory_dir=str(tmp_path / 'mem'))
        text = open(written, encoding='utf-8').read()
        first, _, body = text.partition('\n')
        assert first == 'Rating: 7.0'
        assert 'Rating:' not in body, 'training strips only line 1'

    def test_a_rating_survives_renaming_the_file(self, cfg, media, tmp_path):
        """The memory is keyed by content, so the name is free to change."""
        mem = str(tmp_path / 'mem')
        target = media / 'notes about music.txt'
        api.rate(str(target), 9.0, cfg=cfg, memory_dir=mem)

        renamed = media / 'completely different name.txt'
        target.rename(renamed)

        rows = api.rank_by_rating([str(media)], cfg=cfg, memory_dir=mem)
        assert (f'osfs://{renamed}', 9.0, 'user') in rows

    def test_rating_twice_keeps_the_latest(self, cfg, media, tmp_path):
        import datetime
        mem = str(tmp_path / 'mem')
        target = str(media / 'notes about music.txt')
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        memory.save_rating(target, 3.0, cfg=cfg, memory_dir=mem, when=yesterday)
        memory.save_rating(target, 9.5, cfg=cfg, memory_dir=mem)

        assert memory.load_ratings(mem)[get_file_soft_hash(target)] == 9.5
        assert len(memory.find_memory_files(mem, get_file_soft_hash(target))) == 2, \
            'both ratings are kept as history'

    def test_no_partial_files_left_behind(self, cfg, media, tmp_path):
        mem = str(tmp_path / 'mem')
        api.rate(str(media / 'notes about music.txt'), 5.0, cfg=cfg, memory_dir=mem)
        for root, _dirs, files in os.walk(mem):
            assert not [f for f in files if f.endswith('.partial')], files


class TestRanking:
    def test_unrated_files_are_omitted_without_predicted(self, cfg, media, tmp_path):
        mem = str(tmp_path / 'mem')
        api.rate(str(media / 'notes about music.txt'), 8.0, cfg=cfg, memory_dir=mem)
        rows = api.rank_by_rating([str(media)], cfg=cfg, memory_dir=mem)
        assert len(rows) == 1 and rows[0][2] == 'user'

    def test_best_first(self, cfg, media, tmp_path):
        mem = str(tmp_path / 'mem')
        api.rate(str(media / 'notes about music.txt'), 4.0, cfg=cfg, memory_dir=mem)
        api.rate(str(media / 'music' / 'shopping list.txt'), 9.0, cfg=cfg, memory_dir=mem)
        rows = api.rank_by_rating([str(media)], cfg=cfg, memory_dir=mem)
        assert [r[1] for r in rows] == [9.0, 4.0]

    def test_predicted_says_so_when_there_is_no_trained_model(self, cfg, media, tmp_path):
        mem = str(tmp_path / 'mem')
        with pytest.raises(FileNotFoundError, match='anagnorisis train'):
            api.rank_by_rating([str(media)], cfg=cfg, memory_dir=mem, predicted=True)

    def test_your_rating_wins_over_the_model(self, cfg, media, tmp_path, monkeypatch):
        """The model imitates you; where you have spoken there is nothing to predict."""
        mem = str(tmp_path / 'mem')
        rated = str(media / 'notes about music.txt')
        api.rate(rated, 4.0, cfg=cfg, memory_dir=mem)

        def fake_predict(files, *, cfg, ctx, models_folder=None):
            assert rated not in files, 'a file you rated must not be sent to the model'
            return [(f, 9.9) for f in files]

        monkeypatch.setattr(api, '_predict_ratings', fake_predict)
        rows = api.rank_by_rating([str(media)], cfg=cfg, memory_dir=mem, predicted=True)
        sources = {os.path.basename(p): src for p, _r, src in rows}
        assert sources['notes about music.txt'] == 'user'
        assert sources['shopping list.txt'] == 'model'


class TestQueryIsEmbeddedOnce:
    """Semantic search spans several media types but one vector space.

    Embedding the query per type would multiply the only expensive part of a
    search — and it runs on the CPU, where that cost is most of the wall clock.
    """

    def test_one_embedding_for_a_multi_type_search(self, cfg, tmp_path, monkeypatch):
        media = tmp_path / 'mixed'
        media.mkdir()
        (media / 'a.txt').write_text('text file', encoding='utf-8')
        (media / 'b.jpg').write_bytes(b'\xff\xd8\xff\xe0not-a-real-jpeg')
        (media / 'c.mp3').write_bytes(b'ID3not-a-real-mp3')

        embeds = []

        class FakeEngine:
            def __init__(self, name): self.name = name
            def process_text(self, q):
                embeds.append((self.name, q))
                return [0.5]
            def process_files(self, files, generate_embs_if_not_in_cache=False):
                return [None for _ in files]
            def compare(self, embeddings, query):
                return [0.1 for _ in embeddings]

        monkeypatch.setattr('anagnorisis_core.search.content_search.get_content_search',
                            lambda cfg, name: FakeEngine(name))
        api.search('a quiet street', [str(media)], cfg=cfg, mode='semantic')

        assert len(embeds) == 1, f'query embedded {len(embeds)} times: {embeds}'
