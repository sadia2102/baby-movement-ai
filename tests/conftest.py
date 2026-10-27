import json
from pathlib import Path

import pytest


@pytest.fixture
def document():
    return json.loads((Path(__file__).parent / "fixtures/synthetic_metadata.json").read_text())
