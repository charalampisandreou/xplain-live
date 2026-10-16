"""scripts/deploy_azure.ps1: one region, local build + push, infra/main.json, no credentials, safe what-if."""
import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy_azure.ps1"


def code() -> str:
    """The script without its comment help block and # comments (the help may name what it avoids)."""
    text = re.sub(r"<#.*?#>", "", SCRIPT.read_text(encoding="utf-8"), flags=re.S)
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def test_location_is_germanywestcentral_only():
    assert re.search(r"\[ValidateSet\('germanywestcentral'\)\]", code())
    assert set(re.findall(r"\b[a-z]+(?:west|east|north|south|central)[a-z0-9]*\b", code())) == {"germanywestcentral"}


def test_builds_locally_as_one_amd64_manifest_never_with_acr_tasks():
    src = code()
    assert "docker build --platform linux/amd64 --provenance=false --sbom=false" in src and "docker push" in src
    assert "acr build" not in src and "--source" not in src and not re.search(r"containerapp up\b", src)


def test_no_admin_credentials_tokens_or_passwords():
    src = code()
    for forbidden in ("--admin-enabled true", "acr credential", "acr token", "scope-map", "--expose-token",
                      "--password", "--registry-username", "--secrets", "--logs-workspace-key", "sharedKey"):
        assert forbidden not in src, forbidden


def test_deploys_the_template_with_the_image_and_no_cli_app_path():
    src = code()
    assert "infra/main.json" in src and "deployment group create" in src and "image=$image" in src
    for old in ("containerapp create", "containerapp update", "containerapp registry", "containerapp env create"):
        assert old not in src, old


def test_what_if_prints_resource_ids_only_and_asks_before_deploying():
    src = code()
    what_ifs = [m.start() for m in re.finditer(r"deployment group what-if", src)]
    assert len(what_ifs) == 1
    statement = src[what_ifs[0]:src.index("\n", src.index("--result-format", what_ifs[0]))]
    assert "--result-format ResourceIdOnly" in statement and "--no-pretty-print" not in src
    assert what_ifs[0] < src.index("Read-Host") < src.index("deployment group create")


def test_tag_is_the_git_short_sha_of_a_clean_tree():
    src = code()
    assert "status --porcelain" in src and "rev-parse --short HEAD" in src
