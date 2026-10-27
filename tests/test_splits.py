from dataclasses import asdict, replace
from itertools import combinations

import pytest

from infant_movement.datasets.metadata import PARTITIONS, MetadataError, validate_metadata
from infant_movement.evaluation.splits import GROUP_FIELDS, check_grouped_splits


def test_clean_partitions_pass_without_claiming_patient_independence(document):
    report = check_grouped_splits(validate_metadata(document))
    assert report.ok
    assert report.record_count == 3
    assert report.partition_counts == {"train": 1, "validation": 1, "test": 1}
    assert report.unknown_subject_count == 1
    assert "Unknown subject identity" in report.to_dict()["warnings"][0]
    assert any(
        "does not establish patient independence" in text
        for text in report.to_dict()["limitations"]
    )


@pytest.mark.parametrize("field", GROUP_FIELDS)
@pytest.mark.parametrize(("left", "right"), list(combinations(PARTITIONS, 2)))
def test_every_group_cannot_cross_any_pair_of_partitions(document, field, left, right):
    records = validate_metadata(document)[:2]
    records = [replace(records[0], project_split=left), replace(records[1], project_split=right)]
    records[0] = replace(records[0], **{field: "shared-group"})
    records[1] = replace(records[1], **{field: "shared-group"})
    report = check_grouped_splits(records)
    (issue,) = report.errors
    assert not report.ok
    assert issue.code == f"{field}_overlap"
    assert issue.rows == (1, 2)
    assert issue.partitions == (left, right)


def test_subject_in_different_recordings_is_leakage(document):
    records = list(validate_metadata(document))
    records[1] = replace(records[1], subject_id=records[0].subject_id)
    assert records[0].source_video_id != records[1].source_video_id
    report = check_grouped_splits(records)
    assert [issue.code for issue in report.errors] == ["subject_id_overlap"]


def test_unknown_subjects_do_not_hide_source_video_leakage(document):
    first = replace(validate_metadata(document)[2], project_split="train")
    second = replace(first, sample_id="another-frame", project_split="test")
    report = check_grouped_splits([first, second])
    assert report.unknown_subject_count == 2
    assert {issue.code for issue in report.errors} == {
        "source_video_id_overlap",
        "source_group_id_overlap",
    }


def test_reuploads_with_different_videos_and_groups_are_linked_by_duplicate_registry(document):
    first, second = validate_metadata(document)[:2]
    first = replace(first, duplicate_group_id="known-reupload")
    second = replace(second, duplicate_group_id="known-reupload")
    report = check_grouped_splits([first, second])
    assert [issue.code for issue in report.errors] == ["duplicate_group_id_overlap"]


def test_cross_dataset_and_release_shared_source_is_leakage(document):
    first, second = validate_metadata(document)[:2]
    second = replace(
        second, dataset_id="another-dataset", release_id="v2", source_video_id=first.source_video_id
    )
    report = check_grouped_splits([first, second])
    assert [issue.code for issue in report.errors] == ["source_video_id_overlap"]


def test_transitive_source_and_duplicate_chain_is_not_missed(document):
    first, second, third = validate_metadata(document)
    second = replace(
        second,
        project_split="train",
        source_video_id=first.source_video_id,
        duplicate_group_id="linked-reupload",
    )
    third = replace(third, duplicate_group_id="linked-reupload")
    report = check_grouped_splits([first, second, third])
    (issue,) = report.errors
    assert issue.code == "duplicate_group_id_overlap"
    assert issue.rows == (2, 3)
    assert issue.partitions == ("train", "test")


def test_frames_can_share_groups_inside_one_partition(document):
    first = validate_metadata(document)[0]
    second = replace(first, sample_id="second-frame", frame_index=1, timestamp_seconds=0.04)
    assert check_grouped_splits([first, second]).ok


def test_multiple_unknown_subjects_are_not_fabricated_into_one_group(document):
    first, second, third = validate_metadata(document)
    first = replace(first, subject_id=None, identity_status="unknown", identity_evidence=None)
    second = replace(second, subject_id=None, identity_status="unknown", identity_evidence=None)
    report = check_grouped_splits([first, second, third])
    assert report.ok
    assert report.unknown_subject_count == 3


@pytest.mark.parametrize("partition", ["train", "test"])
def test_repeated_sample_rejected_even_inside_one_partition(document, partition):
    first = validate_metadata(document)[0]
    repeated = replace(first, project_split=partition)
    report = check_grouped_splits([first, repeated])
    assert any(issue.code == "duplicate_sample" for issue in report.errors)


def test_sample_keys_are_scoped_but_provenance_keys_are_global(document):
    first, second = validate_metadata(document)[:2]
    second = replace(second, dataset_id="another-dataset", sample_id=first.sample_id)
    assert check_grouped_splits([first, second]).ok


def test_still_images_without_video_ids_have_explicit_coverage_limit(document):
    record = replace(validate_metadata(document)[2], source_video_id=None)
    report = check_grouped_splits([record])
    assert report.ok
    assert report.missing_source_video_count == 1
    assert "Missing source-video" in report.to_dict()["warnings"][1]


def test_official_splits_are_preserved_and_not_used_for_project_audit(document):
    records = validate_metadata(document)
    before = [asdict(record) for record in records]
    assert check_grouped_splits(records).ok
    assert [asdict(record) for record in records] == before
    assert records[1].official_split != records[1].project_split


@pytest.mark.parametrize("records", [[], [{}]])
def test_empty_or_unvalidated_inputs_do_not_pass(records):
    with pytest.raises(MetadataError):
        check_grouped_splits(records)
