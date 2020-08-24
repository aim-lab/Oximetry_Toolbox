# -*- coding: utf-8 -*-

import pytest

import spo2
from spo2.main import parse_args


class TestParseArgs():

    def setup(self):
        pass

    def test_version(self, capsys):
        with pytest.raises(SystemExit):
            parse_args(str.split("--version"))
        captured = capsys.readouterr()
        assert captured.out.strip() == f"spo2 {spo2.__version__}"
