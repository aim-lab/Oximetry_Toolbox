# -*- coding: utf-8 -*-

import pytest

import OBM
from OBM.main import parse_args


class TestParseArgs():

    def setup(self):
        pass

    def test_version(self, capsys):
        with pytest.raises(SystemExit):
            parse_args(str.split("--version"))
        captured = capsys.readouterr()
        assert captured.out.strip() == f"OBM {OBM.__version__}"
