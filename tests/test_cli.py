import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize(("case", "expected"), [("clean", 0), ("overlap", 1), ("invalid", 2)])
def test_installed_module_reports_exit_status_and_keeps_input_unchanged(
    document, tmp_path, case, expected
):
    if case == "overlap":
        document["records"][1]["subject_id"] = document["records"][0]["subject_id"]
    elif case == "invalid":
        del document["records"][0]["source_group_id"]
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(document))
    before = path.read_bytes()
    result = subprocess.run(
        [sys.executable, "-m", "infant_movement", "audit", str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected
    assert not result.stderr
    assert json.loads(result.stdout)["ok"] is (expected == 0)
    assert "subject-001" not in result.stdout
    assert path.read_bytes() == before


def test_missing_file_returns_json_without_private_path(tmp_path):
    path = tmp_path / "private-file-that-does-not-exist.json"
    result = subprocess.run(
        [sys.executable, "-m", "infant_movement", "audit", str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout) == {"ok": False, "error": "cannot read local manifest"}
    assert str(path) not in result.stdout + result.stderr
