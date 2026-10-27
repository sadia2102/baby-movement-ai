import json
import math
from dataclasses import FrozenInstanceError

import pytest

from infant_movement.datasets.metadata import (
    MetadataError,
    MetadataRecord,
    load_metadata,
    validate_metadata,
)


def test_metadata_roundtrip_and_official_membership(document, tmp_path):
    document["records"][1]["official_split"] = "Original train / development"
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(document))
    records = load_metadata(path)
    assert records[1].official_split == "Original train / development"
    assert records[1].project_split == "validation"
    assert records[2].subject_id is None
    assert records[2].identity_status == "unknown"
    assert records[2].timestamp_seconds is None
    with pytest.raises(FrozenInstanceError):
        records[0].project_split = "test"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_id", ""),
        ("release_id", " v1"),
        ("sample_id", 7),
        ("source_group_id", None),
        ("source_video_id", "https://example.invalid/video"),
        ("source_video_id", "unknown"),
        ("project_split", "val"),
        ("project_split", None),
        ("identity_status", "inferred"),
        ("identity_status", "unknown"),  # A known subject ID is still present.
        ("subject_id", None),
        ("identity_evidence", None),
        ("official_split", ""),
        ("official_split", "   "),
        ("official_split", 1),
        ("frame_index", -1),
        ("frame_index", True),
        ("frame_index", 1.5),
        ("timestamp_seconds", -0.01),
        ("timestamp_seconds", True),
        ("timestamp_seconds", "0.1"),
        ("timestamp_seconds", math.nan),
        ("timestamp_seconds", math.inf),
        ("timestamp_seconds", 10**400),
        ("source_video_id", None),  # The row contains frame/time metadata.
        ("seizure_label", 0),  # No task labels are accepted or synthesized.
    ],
)
def test_invalid_metadata_fails_without_coercion(document, field, value):
    document["records"][0][field] = value
    with pytest.raises(MetadataError, match="record 1"):
        validate_metadata(document)


@pytest.mark.parametrize(
    "field",
    [
        "dataset_id",
        "release_id",
        "sample_id",
        "source_group_id",
        "project_split",
        "identity_status",
    ],
)
def test_missing_required_fields_fail(document, field):
    del document["records"][0][field]
    with pytest.raises(MetadataError, match="missing required fields"):
        validate_metadata(document)


@pytest.mark.parametrize("value", [None, [], {}, {"schema_version": 1, "records": []}])
def test_invalid_or_empty_envelopes_fail(value):
    with pytest.raises(MetadataError):
        validate_metadata(value)


@pytest.mark.parametrize("version", [True, 1.0, "1", 2, None])
def test_unsupported_schema_versions_fail(document, version):
    document["schema_version"] = version
    with pytest.raises(MetadataError, match="schema_version"):
        validate_metadata(document)


def test_nonobject_record_fails(document):
    document["records"][0] = []
    with pytest.raises(MetadataError, match="record 1"):
        validate_metadata(document)


@pytest.mark.parametrize(
    "content",
    [
        b'{"schema_version":1,"schema_version":1,"records":[]}',
        b'{"schema_version":1,"records":[{"sample_id":"a","sample_id":"b"}]}',
        b'{"schema_version":NaN,"records":[]}',
        b'{"schema_version":Infinity,"records":[]}',
        b"invalid JSON",
        b"\xff",
        b'{"schema_version":' + b"1" * 5000 + b',"records":[]}',
    ],
)
def test_malformed_json_fails(tmp_path, content):
    path = tmp_path / "invalid.json"
    path.write_bytes(content)
    with pytest.raises(MetadataError):
        load_metadata(path)


def test_direct_record_construction_validates(document):
    with pytest.raises(MetadataError, match="project_split"):
        MetadataRecord(**{**document["records"][0], "project_split": "training"})
