"""news_viral_pro._load_env_file handles the cases the regular shell would."""
import os
from pathlib import Path

from news_viral_pro import _load_env_file


def test_loads_simple_kv(tmp_path, monkeypatch, clean_env):
    p = tmp_path / ".env"
    p.write_text("FOO=bar\n")
    _load_env_file(p)
    assert os.environ["FOO"] == "bar"


def test_ignores_comments_and_blanks(tmp_path, monkeypatch, clean_env):
    p = tmp_path / ".env"
    p.write_text("# this is a comment\n\nFOO=bar\n# another comment\n")
    _load_env_file(p)
    assert os.environ["FOO"] == "bar"


def test_overrides_empty_env_vars(tmp_path, monkeypatch, clean_env):
    monkeypatch.setenv("FOO", "")
    p = tmp_path / ".env"
    p.write_text("FOO=from_file\n")
    _load_env_file(p)
    assert os.environ["FOO"] == "from_file"


def test_does_not_override_set_env_vars(tmp_path, monkeypatch, clean_env):
    monkeypatch.setenv("FOO", "from_shell")
    p = tmp_path / ".env"
    p.write_text("FOO=from_file\n")
    _load_env_file(p)
    assert os.environ["FOO"] == "from_shell"


def test_value_with_equals_sign_preserved(tmp_path, monkeypatch, clean_env):
    p = tmp_path / ".env"
    p.write_text("URL=https://x.com/path?a=1&b=2\n")
    _load_env_file(p)
    assert os.environ["URL"] == "https://x.com/path?a=1&b=2"


def test_missing_file_is_noop(tmp_path, monkeypatch, clean_env):
    _load_env_file(tmp_path / "does_not_exist.env")
    # No exception, nothing set.
