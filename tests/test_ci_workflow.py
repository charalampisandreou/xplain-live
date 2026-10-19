""".github/workflows/tests.yml: the suite on push and pull request to main, release files checked by SHA-256."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"


def text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_runs_on_push_and_pull_request_to_main_with_python_313():
    src = text()
    assert re.search(r"push:\s*\n\s*branches: \[main\]", src) and re.search(r"pull_request:\s*\n\s*branches: \[main\]", src)
    assert "runs-on: ubuntu-latest" in src and 'python-version: "3.13"' in src and "cache: pip" in src
    assert "pip install -r requirements.txt -r requirements-dev.txt" in src
    assert "run: python -m pytest -q" in src


def test_env_matches_the_local_run_notes():
    src = text()
    for line in ("PYTHONPATH: src", 'PYTHONUTF8: "1"', "PYTHONIOENCODING: utf-8"):
        assert line in src, line


def test_release_files_are_checked_against_one_set_of_shas():
    src = text()
    shas = dict(re.findall(r"^\s*(SHA_\w+): ([0-9a-f]{64})$", src, re.M))
    assert set(shas) == {"SHA_XPASS_JOBLIB", "SHA_FEATURE_LIST", "SHA_BUNDLE_3001"}
    assert "sha256sum --check --strict" in src
    for name, path in (("SHA_XPASS_JOBLIB", "models/xpass_v2/xpass_v2.joblib"),
                       ("SHA_FEATURE_LIST", "models/xpass_v2/feature_list.json"),
                       ("SHA_BUNDLE_3001", "data/bundle_seed3001.json")):
        assert f"${name}  {path}" in src, name
    assert len(re.findall(r"\b[0-9a-f]{64}\b", src)) == 3          # no second copy of a SHA


def test_bundle_sha_equals_the_one_the_readme_names():
    sha = re.search(r"SHA_BUNDLE_3001: ([0-9a-f]{64})", text()).group(1)
    assert sha in (ROOT / "README.md").read_text(encoding="utf-8")
