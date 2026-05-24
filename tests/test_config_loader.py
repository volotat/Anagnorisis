"""
Tests for src/config_loader.py

Covers:
  - ${VAR:-default} substitution (env var set / unset)
  - $VAR substitution (plain, no default)
  - Missing env var with no default returns empty string
  - Nested values survive YAML parsing
  - Invalid YAML raises an error
  - Integer / bool values remain typed after substitution
"""
import os
import textwrap
import tempfile
import pytest
from src.config_loader import load_config


def _write_config(content: str) -> str:
    """Write content to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    f.write(textwrap.dedent(content))
    f.flush()
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# ${VAR:-default} syntax
# ---------------------------------------------------------------------------

def test_env_var_with_default_uses_env_when_set(monkeypatch):
    monkeypatch.setenv('MY_PATH', '/custom/path')
    cfg_file = _write_config("""
        main:
          data_path: ${MY_PATH:-/default/path}
    """)
    cfg = load_config(cfg_file)
    assert cfg['main']['data_path'] == '/custom/path'


def test_env_var_with_default_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv('MY_PATH', raising=False)
    cfg_file = _write_config("""
        main:
          data_path: ${MY_PATH:-/default/path}
    """)
    cfg = load_config(cfg_file)
    assert cfg['main']['data_path'] == '/default/path'


def test_env_var_default_can_be_empty_string(monkeypatch):
    monkeypatch.delenv('MISSING_VAR', raising=False)
    cfg_file = _write_config("""
        section:
          value: ${MISSING_VAR:-}
    """)
    cfg = load_config(cfg_file)
    assert cfg['section']['value'] == '' or cfg['section']['value'] is None


# ---------------------------------------------------------------------------
# $VAR syntax (plain, no default)
# ---------------------------------------------------------------------------

def test_plain_env_var_substituted(monkeypatch):
    monkeypatch.setenv('PORT', '8080')
    cfg_file = _write_config("""
        server:
          port: $PORT
    """)
    cfg = load_config(cfg_file)
    # YAML will parse "8080" as int
    assert str(cfg['server']['port']) == '8080'


def test_plain_env_var_missing_returns_empty(monkeypatch):
    monkeypatch.delenv('UNDEFINED_VAR_XYZ', raising=False)
    cfg_file = _write_config("""
        section:
          key: $UNDEFINED_VAR_XYZ
    """)
    cfg = load_config(cfg_file)
    # An empty string in YAML is parsed as None
    assert cfg['section']['key'] is None or cfg['section']['key'] == ''


# ---------------------------------------------------------------------------
# Nested structure integrity
# ---------------------------------------------------------------------------

def test_nested_yaml_values_preserved(monkeypatch):
    monkeypatch.setenv('HOST', '0.0.0.0')
    cfg_file = _write_config("""
        main:
          host: ${HOST:-localhost}
          port: 5001
          nested:
            flag: true
            items:
              - one
              - two
    """)
    cfg = load_config(cfg_file)
    assert cfg['main']['host'] == '0.0.0.0'
    assert cfg['main']['port'] == 5001
    assert cfg['main']['nested']['flag'] is True
    assert cfg['main']['nested']['items'] == ['one', 'two']


# ---------------------------------------------------------------------------
# Invalid YAML
# ---------------------------------------------------------------------------

def test_invalid_yaml_raises():
    cfg_file = _write_config("""
        key: [unclosed bracket
    """)
    with pytest.raises(Exception):
        load_config(cfg_file)


# ---------------------------------------------------------------------------
# Multiple substitutions in same file
# ---------------------------------------------------------------------------

def test_multiple_substitutions_in_same_file(monkeypatch):
    monkeypatch.setenv('IMG_PATH', '/images')
    monkeypatch.setenv('MUSIC_PATH', '/music')
    monkeypatch.delenv('TEXT_PATH', raising=False)
    cfg_file = _write_config("""
        images:
          media_directory: ${IMG_PATH:-/default_img}
        music:
          media_directory: ${MUSIC_PATH:-/default_music}
        text:
          media_directory: ${TEXT_PATH:-/default_text}
    """)
    cfg = load_config(cfg_file)
    assert cfg['images']['media_directory'] == '/images'
    assert cfg['music']['media_directory'] == '/music'
    assert cfg['text']['media_directory'] == '/default_text'


if __name__ == '__main__':
    import sys
    print("Running config_loader tests...")
    pytest.main([__file__, '-v'])
