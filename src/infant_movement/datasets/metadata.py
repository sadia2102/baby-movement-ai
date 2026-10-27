"""Strict, label-free metadata envelope for local provenance audits.

IDs are pseudonymous registry keys, never inferred identities. Shared entities
must use the same registry key across datasets and releases. No assets are read.
"""

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

PARTITIONS = ("train", "validation", "test")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_UNKNOWN = {"unknown", "unverified", "none", "null", "na", "n/a"}


class MetadataError(ValueError):
    """The manifest cannot safely be interpreted under schema version 1."""


def _identifier(name: str, value: object, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not _ID.fullmatch(value) or value.lower() in _UNKNOWN:
        raise MetadataError(
            f"{name} must be a pseudonymous registry key" + (" or null" if nullable else "")
        )


@dataclass(frozen=True, kw_only=True)
class MetadataRecord:
    """One metadata record, with explicit unknown identity and project partition.

    This is the M1 subset of the planned sample envelope. Labels, image paths,
    coordinate payloads, and model features belong to future separate adapters.
    """

    dataset_id: str
    release_id: str
    sample_id: str
    source_group_id: str
    project_split: str
    identity_status: str
    subject_id: str | None = None
    identity_evidence: str | None = None
    session_id: str | None = None
    source_video_id: str | None = None
    duplicate_group_id: str | None = None
    official_split: str | None = None
    frame_index: int | None = None
    timestamp_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("dataset_id", "release_id", "sample_id", "source_group_id"):
            _identifier(name, getattr(self, name))
        for name in (
            "subject_id",
            "identity_evidence",
            "session_id",
            "source_video_id",
            "duplicate_group_id",
        ):
            _identifier(name, getattr(self, name), nullable=True)
        if self.official_split is not None and (
            not isinstance(self.official_split, str) or not self.official_split.strip()
        ):
            raise MetadataError("official_split must be a nonempty original name or null")
        if self.project_split not in PARTITIONS:
            raise MetadataError("project_split must be train, validation, or test")
        if self.identity_status not in ("known", "unknown"):
            raise MetadataError("identity_status must be known or unknown")
        if self.identity_status == "known" and (
            self.subject_id is None or self.identity_evidence is None
        ):
            raise MetadataError("known identity requires subject_id and identity_evidence")
        if self.identity_status == "unknown" and self.subject_id is not None:
            raise MetadataError("unknown identity requires subject_id=null")
        if self.frame_index is not None and (
            type(self.frame_index) is not int or self.frame_index < 0
        ):
            raise MetadataError("frame_index must be a nonnegative integer or null")
        if self.timestamp_seconds is not None:
            if type(self.timestamp_seconds) not in (int, float):
                raise MetadataError("timestamp_seconds must be a finite nonnegative number or null")
            try:
                valid_time = math.isfinite(self.timestamp_seconds) and self.timestamp_seconds >= 0
            except OverflowError:
                valid_time = False
            if not valid_time:
                raise MetadataError("timestamp_seconds must be a finite nonnegative number or null")
        if (self.frame_index is not None or self.timestamp_seconds is not None) and (
            self.source_video_id is None
        ):
            raise MetadataError("frame/time metadata requires source_video_id")

    @classmethod
    def from_mapping(cls, value: object) -> "MetadataRecord":
        if not isinstance(value, Mapping):
            raise MetadataError("each record must be a JSON object")
        allowed = {field.name for field in fields(cls)}
        if set(value) - allowed:
            raise MetadataError("record contains unsupported fields; use metadata only")
        required = {
            "dataset_id",
            "release_id",
            "sample_id",
            "source_group_id",
            "project_split",
            "identity_status",
        }
        missing = required - set(value)
        if missing:
            raise MetadataError("missing required fields: " + ", ".join(sorted(missing)))
        return cls(**value)


def validate_metadata(document: object) -> tuple[MetadataRecord, ...]:
    """Validate a complete versioned manifest, without coercing or filling labels."""
    if not isinstance(document, dict) or set(document) != {"schema_version", "records"}:
        raise MetadataError("manifest requires exactly schema_version and records")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise MetadataError("only integer schema_version=1 is supported")
    values = document["records"]
    if not isinstance(values, list) or not values:
        raise MetadataError("records must be a nonempty JSON array")
    records = []
    for row, value in enumerate(values, start=1):
        try:
            records.append(MetadataRecord.from_mapping(value))
        except MetadataError as error:
            raise MetadataError(f"record {row}: {error}") from error
    return tuple(records)


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise MetadataError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise MetadataError("nonfinite JSON constants are not supported")


def load_metadata(path: str | Path) -> tuple[MetadataRecord, ...]:
    """Load a local JSON file. No URLs, data assets, or network calls are used."""
    try:
        with Path(path).open(encoding="utf-8") as stream:
            document = json.load(
                stream, object_pairs_hook=_unique_keys, parse_constant=_reject_constant
            )
    except MetadataError:
        raise
    except (ValueError, UnicodeError) as error:
        raise MetadataError("manifest must be valid UTF-8 JSON") from error
    return validate_metadata(document)
