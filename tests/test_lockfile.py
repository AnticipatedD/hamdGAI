# tests/test_lockfile.py
from pathlib import Path

def test_requirements_lock_exists_and_non_empty():
    lock = Path("requirements-lock.txt")
    assert lock.is_file(), "requirements-lock.txt is missing"
    assert lock.stat().st_size > 0, "requirements-lock.txt is empty"
