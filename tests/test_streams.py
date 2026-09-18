import sys

import main


def test_setup_std_streams_noop_when_streams_present():
    before_out, before_err = sys.stdout, sys.stderr
    assert main.setup_std_streams() is None
    assert sys.stdout is before_out
    assert sys.stderr is before_err


def test_setup_std_streams_redirects_none_streams(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'stdout', None)
    monkeypatch.setattr(sys, 'stderr', None)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    monkeypatch.setattr(sys, 'platform', 'win32')

    stream = main.setup_std_streams()
    try:
        assert sys.stdout is not None
        assert sys.stderr is not None
        sys.stdout.flush()
        print('x')
        assert (tmp_path / 'ERA' / 'era.log').exists()
    finally:
        if stream is not None:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            stream.close()
