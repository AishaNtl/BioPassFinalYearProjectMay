# end to end authorisation test

import pytest
from biopass.add_faces3 import start_authentication

@pytest.mark.integration
class TestFullAuthentication:
    @pytest.fixture(autouse=True)
    def mock_user_interaction(self, monkeypatch):
        """Simulate user performing correct actions"""
        monkeypatch.setattr('builtins.input', lambda _: None)
        monkeypatch.setattr('add_faces3.blink_detection', lambda _: True)
        monkeypatch.setattr('add_faces3.detect_expression_with_timer', lambda *_: True)

    def test_full_authentication_flow(self):
        assert start_authentication() is True

    def test_failed_authentication(self, monkeypatch):
        monkeypatch.setattr('add_faces3.detect_expression_with_timer', lambda *_: False)
        assert start_authentication() is False