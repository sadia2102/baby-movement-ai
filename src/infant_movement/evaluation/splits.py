"""Audit supplied assignments before extracting frames or fitting any features."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from infant_movement.datasets.metadata import PARTITIONS, MetadataError, MetadataRecord

GROUP_FIELDS = ("subject_id", "source_video_id", "source_group_id", "duplicate_group_id")
LIMITATIONS = (
    "Only supplied registry keys are checked; identities, reuploads, near duplicates, "
    "rights, and upstream training membership are not independently verified.",
    "A passing audit does not establish patient independence or clinical validity.",
)


@dataclass(frozen=True)
class SplitIssue:
    code: str
    rows: tuple[int, ...]
    partitions: tuple[str, ...]


@dataclass(frozen=True)
class SplitReport:
    record_count: int
    partition_counts: dict[str, int]
    unknown_subject_count: int
    missing_source_video_count: int
    errors: tuple[SplitIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready counts and row references, without registry IDs."""
        return {
            "schema_version": 1,
            "ok": self.ok,
            **asdict(self),
            "warnings": [
                message
                for count, message in (
                    (
                        self.unknown_subject_count,
                        "Unknown subject identity limits independence claims.",
                    ),
                    (
                        self.missing_source_video_count,
                        "Missing source-video IDs limit video audits.",
                    ),
                )
                if count
            ],
            "limitations": LIMITATIONS,
        }


def check_grouped_splits(records: Iterable[MetadataRecord]) -> SplitReport:
    """Check every known group across all three partitions, across datasets too.

    A shared registry may contain multiple datasets' metadata, never their labels.
    Null IDs are skipped, not treated as a shared invented identity. Official
    splits are reference metadata and are never changed or used as assignments.
    Call separately for each CV fold/protocol, with one assignment per sample.
    """
    records = tuple(records)
    if not records:
        raise MetadataError("cannot audit an empty manifest")
    if any(not isinstance(record, MetadataRecord) for record in records):
        raise MetadataError("split checks require validated MetadataRecord objects")
    counts = dict.fromkeys(PARTITIONS, 0)
    sample_rows = defaultdict(list)
    groups = {field: defaultdict(list) for field in GROUP_FIELDS}
    for row, record in enumerate(records, start=1):
        counts[record.project_split] += 1
        sample_rows[(record.dataset_id, record.release_id, record.sample_id)].append(row)
        for field in GROUP_FIELDS:
            value = getattr(record, field)
            if value is not None:
                groups[field][value].append(row)

    def partitions(rows: list[int]) -> tuple[str, ...]:
        present = {records[row - 1].project_split for row in rows}
        return tuple(partition for partition in PARTITIONS if partition in present)

    errors = []
    for rows in sample_rows.values():
        if len(rows) > 1:
            errors.append(SplitIssue("duplicate_sample", tuple(rows), partitions(rows)))
    for field, registry in groups.items():
        for rows in registry.values():
            assigned = partitions(rows)
            if len(assigned) > 1:
                errors.append(SplitIssue(f"{field}_overlap", tuple(rows), assigned))
    return SplitReport(
        record_count=len(records),
        partition_counts=counts,
        unknown_subject_count=sum(record.subject_id is None for record in records),
        missing_source_video_count=sum(record.source_video_id is None for record in records),
        errors=tuple(errors),
    )
