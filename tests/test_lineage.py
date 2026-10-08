
import pytest

from mini_mlops.lineage import sha256_file


def test_sha256_known_content(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_bytes(b"abc")

    result = sha256_file(path)

    assert result == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def test_same_content_same_hash(tmp_path):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"

    file_a.write_bytes(b"same content")
    file_b.write_bytes(b"same content")

    assert sha256_file(file_a) == sha256_file(file_b)


def test_content_change_changes_hash(tmp_path):
    path = tmp_path / "data.csv"

    path.write_bytes(b"version 1")
    hash_v1 = sha256_file(path)

    path.write_bytes(b"version 2")
    hash_v2 = sha256_file(path)

    assert hash_v1 != hash_v2


def test_missing_file_raises(tmp_path):
    path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        sha256_file(path)
