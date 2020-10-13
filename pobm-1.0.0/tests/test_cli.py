# -*- coding: utf-8 -*-

import pytest

import pobm
from pobm.main import parse_args


class TestParseArgs():

    def setup(self):
        pass

    def test_version(self, capsys):
        with pytest.raises(SystemExit):
            parse_args(str.split("--version"))
        captured = capsys.readouterr()
        assert captured.out.strip() == f"pobm {pobm.__version__}"
