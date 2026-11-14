"""Keep Settings, .env.example and a local .env in step, so a new setting is never
documented in one place and missing from another."""

import re
from pathlib import Path

import pytest

from retail_rag.config import Settings

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / ".env.example"

# read by az commands, Bicep or Azure Monitor, not by Settings
DEPLOYMENT_VARS = {
    "AZURE_LOCATION",
    "AZURE_SEARCH_LOCATION",
    "AZURE_RESOURCE_GROUP",
    "AZURE_PREFIX",
}
# paths with sensible defaults inside the repo / container; not meant for .env
INTERNAL_SETTINGS = {"DATA_DIR", "PROMPT_PATH", "GOLDEN_PATH"}


def names(path: Path) -> list[str]:
    return re.findall(r"^([A-Z_][A-Z0-9_]*)=", path.read_text(encoding="utf-8"), re.M)


def test_example_documents_every_setting():
    settings = {f.upper() for f in Settings.model_fields} - INTERNAL_SETTINGS
    missing = settings - set(names(EXAMPLE))
    assert not missing, f"add to .env.example: {sorted(missing)}"


def test_example_has_no_unknown_or_duplicate_variables():
    example = names(EXAMPLE)
    known = {f.upper() for f in Settings.model_fields} | DEPLOYMENT_VARS
    assert not set(example) - known, f"not read anywhere: {sorted(set(example) - known)}"
    assert len(example) == len(set(example)), "duplicate variables in .env.example"


def test_local_env_matches_example():
    env = ROOT / ".env"
    if not env.exists():
        pytest.skip("no local .env (e.g. in CI)")
    local, example = set(names(env)), set(names(EXAMPLE))
    assert local == example, (
        f"missing from .env: {sorted(example - local)}; not in .env.example: "
        f"{sorted(local - example)}"
    )
