"""Descriptions the model wrote without ever receiving the media.

Roughly a third of one audio library came out like this. It is worse than a
plain failure: the text is fluent, plausible, and permanent once it reaches a
.meta file, which nothing ever overwrites. So it has to be recognised and
treated as the failure it is.

Every positive case below is a real string taken from a sidecar. Every negative
case is a real description from the same library — which is the point: the
patterns have to separate them.
"""
import pytest

from anagnorisis_core.models.descriptor import looks_like_a_refusal


REFUSALS = [
    # what _synthesise produces when handed five refusals
    'The audio consists entirely of requests for an audio file or link so that a '
    'description can be provided.',
    'The audio consists entirely of requests for an audio file to be provided so '
    'that a description can be made.',
    'The audio consists of a series of requests for audio files to be provided '
    'for description.',
    'The audio consists primarily of requests for audio input, asking the user to '
    'provide an audio file.',
    'The audio primarily consists of requests for audio files or links, '
    'indicating a process where the system is waiting for input.',
    # single-turn refusals
    'Please provide the audio so I can describe it for you.',
    'Please provide the audio file you are referring to.',
    'Please provide the audio.',
    'I need an audio file to provide a description.',
    'No audio was provided, so I cannot write a description.',
    'Therefore, no description of the audio as a whole can be provided.',
    # the model reading its own placeholder tokens
    'The provided text consists entirely of the letter "A" repeated many times, '
    'which is non-musical.',
    'The audio consists of a sequence of the letter "A" repeated many times, '
    'resulting in a purely textual audio.',
]

REAL_DESCRIPTIONS = [
    # audio that actually worked
    'The audio features a clear, relatively fast-paced melody played on a bright, '
    'plucked string instrument, likely a guitar or mandolin, with a light, '
    'rhythmic accompaniment.',
    'The audio consists of a series of spoken phrases, likely a conversation or a '
    'recording of someone speaking, with some instances of hesitation or apology.',
    'The audio appears to be a continuous recording, likely a conversation or '
    'speech, with varying levels of background noise.',
    # text and image descriptions that mention "provide" innocently
    'The provided text is a highly complex, nested mathematical expression.',
    'The provided text is a complex mathematical expression, not a standard piece '
    'of prose.',
    '*   The white areas provide high contrast, separating the colored lines.',
    'While current AI struggles with the vast context required for open-ended '
    'RPGs, the author suggests a narrower scope.',
    # plausible near-misses worth being explicit about
    'A radio host requests listener call-ins between songs.',
    'The lyrics describe someone unable to let go of a memory.',
    'The image shows a file cabinet beside a link fence.',
]


@pytest.mark.parametrize('text', REFUSALS)
def test_refusals_are_recognised(text):
    assert looks_like_a_refusal(text), text


@pytest.mark.parametrize('text', REAL_DESCRIPTIONS)
def test_real_descriptions_are_kept(text):
    assert not looks_like_a_refusal(text), text


def test_empty_is_not_a_refusal():
    """It is already a failure; the caller does not need it relabelled."""
    assert not looks_like_a_refusal('')
    assert not looks_like_a_refusal(None)


class TestItReachesTheRightBehaviour:
    """The detector is only useful if a refusal ends up as a failure, and a
    failure ends up as no sidecar."""

    def test_a_vetted_description_makes_describe_write_nothing(
            self, tmp_path, monkeypatch):
        import os
        from omegaconf import OmegaConf
        from anagnorisis_core import api
        from anagnorisis_core.media.media_types import bundled_media_types_dir
        from anagnorisis_core.storage.sinks import MetaSink

        conf = OmegaConf.load(os.path.join(bundled_media_types_dir(),
                                           'media_types.yaml'))
        conf.main = {'cache_path': str(tmp_path / 'cache'),
                     'embedding_models_path': str(tmp_path / 'models'),
                     'media_types_path': bundled_media_types_dir()}
        conf.omni = {'model_name': 'x'}
        conf.embedder = {'model_name': 'y'}

        from anagnorisis_core.media import description

        media = tmp_path / 'media'
        media.mkdir()
        # Long enough that the descriptor is actually consulted. A short text
        # file is kept verbatim and never reaches a model, so it could not
        # exercise a refusal.
        (media / 'note.txt').write_text(
            'some words ' * (description.text_verbatim_limit(conf) // 10))

        class RefusingDescriptor:
            def __init__(self, cfg): pass
            def initiate(self, models_folder): pass
            def unload(self): pass
            def describe_text(self, text, prompt=None):
                # what _generate returns once it has vetted a refusal
                return ''

        monkeypatch.setattr(api, 'OmniDescriptor', RefusingDescriptor)
        monkeypatch.setattr(api, '_embed_batch', lambda *a, **k: None)
        monkeypatch.setattr(api, 'get_omni_embedder',
                            lambda cfg: type('E', (), {'unload': lambda self: None})())

        report = api.describe([str(media)], cfg=conf, sink=MetaSink(conf))

        assert not (media / 'note.txt.meta').exists(), (
            'a sidecar was written for a description that was thrown away')
        assert report.written == 0 and report.failed == 1


class TestTheAudioWindow:
    """Gemma 4's audio input is a fixed 30-second window, not a maximum.

    A shorter clip yields proportionally fewer soft tokens — 10 seconds gives 250
    of the 750 the model expects — and the model then does not register them as
    audio at all, answering "Please provide the audio you would like me to
    describe". No error is raised anywhere; the refusal is the only symptom, and
    it cost every audio description in one library.
    """

    def test_a_short_clip_is_padded_to_the_window(self):
        import numpy as np
        from anagnorisis_core.models.descriptor import (AUDIO_WINDOW_SAMPLES,
                                                        fit_audio_window)
        out = fit_audio_window(np.ones(16000 * 10, dtype=np.float32))
        assert len(out) == AUDIO_WINDOW_SAMPLES

    def test_the_padding_is_silence_after_the_real_audio(self):
        """So a 5 second clip describes those 5 seconds, not 30 of noise."""
        import numpy as np
        from anagnorisis_core.models.descriptor import fit_audio_window
        out = fit_audio_window(np.ones(16000 * 5, dtype=np.float32))
        assert out[:16000 * 5].all()
        assert not out[16000 * 5:].any()

    def test_an_over_long_clip_is_trimmed(self):
        import numpy as np
        from anagnorisis_core.models.descriptor import (AUDIO_WINDOW_SAMPLES,
                                                        fit_audio_window)
        out = fit_audio_window(np.ones(16000 * 45, dtype=np.float32))
        assert len(out) == AUDIO_WINDOW_SAMPLES

    def test_an_exact_clip_is_left_alone(self):
        import numpy as np
        from anagnorisis_core.models.descriptor import (AUDIO_WINDOW_SAMPLES,
                                                        fit_audio_window)
        clip = np.ones(AUDIO_WINDOW_SAMPLES, dtype=np.float32)
        assert len(fit_audio_window(clip)) == AUDIO_WINDOW_SAMPLES

    def test_the_window_matches_the_processor_configuration(self):
        """750 soft tokens x 40 ms. If the model is swapped for one with a
        different audio length, this is the assertion that should fail rather
        than every audio file quietly refusing."""
        import json
        import os

        from omegaconf import OmegaConf
        from anagnorisis_core.models.descriptor import AUDIO_WINDOW_SECONDS

        cfg_path = os.path.join('models', 'google__gemma-4-E2B-it',
                                'processor_config.json')
        if not os.path.isfile(cfg_path):
            import pytest
            pytest.skip('model weights not present')
        conf = json.load(open(cfg_path))
        expected = conf['audio_seq_length'] * conf['audio_ms_per_token'] / 1000
        assert AUDIO_WINDOW_SECONDS == expected, (
            f'window is {AUDIO_WINDOW_SECONDS}s but the processor wants {expected}s')

    def test_sampling_asks_for_the_whole_window(self):
        """The proxy passes this to the worker explicitly, so its default is the
        one that decides — the worker's own default never gets a say."""
        import inspect
        from anagnorisis_core.models.descriptor import (AUDIO_WINDOW_SECONDS,
                                                        OmniDescriptor)
        sig = inspect.signature(OmniDescriptor.describe_audio_sampled)
        assert sig.parameters['sample_duration_s'].default == AUDIO_WINDOW_SECONDS


class TestVideoFramesPerWindow:
    """Four stills per window, and it is a measurement rather than a guess.

    Peak VRAM alongside a full audio window on an 8 GB card: 4 stills 7.47 GB,
    6 stills 7.76 GB (about 20 MB spare, and the same description word for word),
    8 stills out of memory. Four keeps the headroom that lets a background pass
    survive something else touching the card.
    """

    def test_the_default_is_four(self):
        import inspect
        from anagnorisis_core.models.descriptor import OmniDescriptor
        sig = inspect.signature(OmniDescriptor.describe_video_sampled)
        assert sig.parameters['frames_per_segment'].default == 4

    def test_the_shipped_config_agrees(self):
        import os
        from omegaconf import OmegaConf
        from anagnorisis_core.config import bundled_defaults_path
        conf = OmegaConf.load(bundled_defaults_path())
        assert conf.omni.video_frames_per_window == 4

    def test_the_config_overrides_the_argument(self, tmp_path):
        """A larger card should be able to raise it without patching code."""
        from omegaconf import OmegaConf
        from anagnorisis_core.models.descriptor import _OmniDescriptorImpl

        class Probe(_OmniDescriptorImpl):
            def __init__(self, cfg):
                self.cfg = cfg

        cfg = OmegaConf.create({'omni': {'video_frames_per_window': 6}})
        probe = Probe(cfg)
        assert int(probe.cfg.omni.get('video_frames_per_window', 4)) == 6
