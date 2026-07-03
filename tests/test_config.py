from src.utils.config import get_settings, Settings


def test_settings_loads():
    s = get_settings()
    assert isinstance(s, Settings)
    assert s.CLAUDE_MODEL
    assert s.CHUNK_SIZE > 0


def test_assert_api_key_passes_with_test_key():
    s = get_settings()
    # In conftest.py we set a test key, so this should not raise.
    s.assert_api_key()
