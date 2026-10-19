"""infra/main.json: no secrets, one region, no Express mode set, identity-based image pull."""
import json
import re
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "infra" / "main.json"


def template() -> dict:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def resources(kind: str) -> list[dict]:
    return [r for r in template()["resources"] if r["type"] == kind]


def walk(node, path=""):
    """Yield (path, key, value) for every key in the template."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield path, k, v
            yield from walk(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}/{i}")


def test_no_secrets_in_the_template():
    text = TEMPLATE.read_text(encoding="utf-8")
    for word in ("password", "passwordSecretRef", "secrets", "connectionString", "registryCredentials"):
        assert word not in text, word
    shared = [v for _, k, v in walk(template()) if k == "sharedKey"]
    assert shared == ["[listKeys(variables('workspaceId'), '2023-09-01').primarySharedKey]"]  # read in ARM only
    assert all(p["type"] == "string" for p in template()["parameters"].values())
    assert not any(k == "sharedKey" for _, k, _ in walk(template().get("outputs", {})))


def test_only_germanywestcentral():
    location = template()["parameters"]["location"]
    assert location["defaultValue"] == "germanywestcentral" and location["allowedValues"] == ["germanywestcentral"]
    text = TEMPLATE.read_text(encoding="utf-8")
    assert set(re.findall(r"\b[a-z]+(?:west|east|north|south|central)[a-z0-9]*\b", text)) == {"germanywestcentral"}
    assert all(r["location"] == "[parameters('location')]" for r in template()["resources"])


def test_environment_mode_is_never_set():
    assert not any(k == "environmentMode" for _, k, _ in walk(template()))
    assert "Express" not in TEMPLATE.read_text(encoding="utf-8")


def test_registry_pull_uses_the_user_assigned_identity():
    (app,) = resources("Microsoft.App/containerApps")
    identity = "[variables('identityId')]"
    assert app["identity"]["type"] == "UserAssigned" and list(app["identity"]["userAssignedIdentities"]) == [identity]
    registries = app["properties"]["configuration"]["registries"]
    assert registries and all(r == {"server": r["server"], "identity": identity} for r in registries)


def test_scale_to_zero_and_size():
    (app,) = resources("Microsoft.App/containerApps")
    assert app["properties"]["template"]["scale"] == {"minReplicas": 0, "maxReplicas": 1}
    assert app["properties"]["configuration"]["ingress"]["targetPort"] == 8000


def test_the_container_gets_the_live_settings_and_the_identity_client_id():
    (app,) = resources("Microsoft.App/containerApps")
    (container,) = app["properties"]["template"]["containers"]
    env = {e["name"]: e["value"] for e in container["env"]}
    assert env == {
        "AZURE_CLIENT_ID": "[reference(variables('identityId'), '2023-01-31').clientId]",   # read, not hard-coded
        "XPLAIN_LIVE": "1",
        "XPLAIN_FOUNDRY_ENDPOINT": "[parameters('foundryEndpoint')]",
        "XPLAIN_LIVE_DEPLOYMENT": "[parameters('liveDeployment')]",
    }
    params = template()["parameters"]
    assert params["foundryEndpoint"]["defaultValue"].endswith(".openai.azure.com/openai/v1/")
    assert params["liveDeployment"]["defaultValue"] == "gpt-4.1-mini"
    assert not any("secretRef" in e for e in container["env"])
    text = TEMPLATE.read_text(encoding="utf-8")
    assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", text)   # no GUIDs
    # the same identity serves the image pull and the app at run time (userAssignedIdentities)
    assert list(app["identity"]["userAssignedIdentities"]) == ["[variables('identityId')]"]
