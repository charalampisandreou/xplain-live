"""requirements-runtime.txt (what the Docker image installs) must use the pins of requirements.txt."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pins(name: str) -> dict[str, str]:
    out = {}
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            package, version = line.split("==")
            out[package.lower()] = version
    return out


def test_runtime_pins_equal_the_full_pins():
    runtime, full = pins("requirements-runtime.txt"), pins("requirements.txt")
    assert runtime, "requirements-runtime.txt has no pins"
    assert {p: v for p, v in runtime.items() if full.get(p) != v} == {}, "a runtime pin differs from requirements.txt"


def test_runtime_has_the_live_client_and_only_the_two_agent_framework_packages():
    runtime = pins("requirements-runtime.txt")
    assert {"openai", "azure-identity"} <= set(runtime)   # src/xplain/live.py: /api/live/health
    # src/xplain/retake.py: the Narrator agent; never the "agent-framework" meta package
    assert {p for p in runtime if p.startswith("agent-framework")} == {"agent-framework-core", "agent-framework-openai"}
