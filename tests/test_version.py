import os

import main


def test_read_version_returns_version_file_content():
    with open(os.path.join(main.root_folder, 'VERSION'), encoding='utf-8') as f:
        expected = f.read().strip()
    assert main.read_version() == expected
    assert main.APP_VERSION == expected


def test_read_version_missing_file_returns_dev(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'root_folder', str(tmp_path))
    assert main.read_version() == 'dev'
