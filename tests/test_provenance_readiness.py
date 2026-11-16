"""Synthetic metadata only: no images, archives, models or training fixtures."""

import copy
import json
import subprocess
import sys

import pytest

from infant_movement.evaluation.provenance_readiness import (
    RELATIONSHIPS,
    aggregate_review,
    canonical_hash,
    classify_claims,
    evidence_matrix,
    frozen_cohort,
    locked_pins,
    match_cohort,
    metadata_path,
    new_review_directory,
    relevant_decisions,
)


@pytest.fixture
def cohort():
    rows = [
        {
            "image_set": "ads",
            "filename": f"synthetic-{i}.jpg",
            "source_row": i + 1,
            "official_partition": "infanface-train",
            "eligible": True,
            "valid_points": 68,
            "subject_id": None,
            "source_video_id": None,
            "source_group_id": None,
        }
        for i in range(32)
    ]
    refs = [
        {k: r[k] for k in ("image_set", "filename", "source_row", "official_partition")}
        for r in rows
    ]
    sha = canonical_hash(refs)
    return {"records": rows, "selection_sha256": sha}, {"selection_sha256": sha}


def test_exact_frozen_order_is_preserved_and_never_reranked(cohort):
    selection, prepared = cohort
    rows = frozen_cohort(selection, prepared)
    assert rows is selection["records"]
    changed = copy.deepcopy(selection)
    changed["records"].reverse()
    with pytest.raises(ValueError):
        frozen_cohort(changed, prepared)


@pytest.mark.parametrize("violation", ["size", "duplicate", "test", "batch", "unusable"])
def test_changed_or_unauthorized_cohort_rejected(cohort, violation):
    selection, prepared = copy.deepcopy(cohort)
    if violation == "size":
        selection["records"].pop()
    elif violation == "duplicate":
        selection["records"][1] = selection["records"][0]
    elif violation == "test":
        selection["records"][0]["official_partition"] = "infanface-test"
    elif violation == "batch":
        selection["records"][0]["image_set"] = "google"
    else:
        selection["records"][0]["eligible"] = False
    with pytest.raises(ValueError):
        frozen_cohort(selection, prepared)


def manufactured_records(rows):
    return {
        f"key-{i}": {
            "official_partition": "infanface-train",
            "archive_image_path": f"infanface-public/images/ads/{r['filename']}",
            "source_row": r["source_row"],
            "subject_id": None,
            "source_video_id": None,
            "source_group_id": None,
        }
        for i, r in enumerate(rows)
    }


class SealedTestRecord(dict):
    def __getitem__(self, key):
        if key != "official_partition":
            pytest.fail("test sample evidence must not be inspected")
        return "infanface-test"


def test_membership_filter_precedes_reference_and_evidence_access(cohort):
    rows = cohort[0]["records"]
    records = {"sealed": SealedTestRecord(), **manufactured_records(rows)}
    keys = match_cohort(rows, records)
    assert keys == [f"key-{i}" for i in range(32)]
    selected = relevant_decisions(keys, records, [{"left": keys[0], "right": "sealed"}], [])
    assert selected == []


@pytest.mark.parametrize("violation", ["missing", "duplicate", "source_row"])
def test_changed_historical_join_rejected(cohort, violation):
    rows = cohort[0]["records"]
    records = manufactured_records(rows)
    if violation == "missing":
        del records["key-0"]
    elif violation == "duplicate":
        records["extra"] = records["key-0"].copy()
    else:
        records["key-0"]["source_row"] += 1
    with pytest.raises(ValueError):
        match_cohort(rows, records)


def test_exact_candidate_endpoints_not_absent_optional_ids(cohort):
    rows = cohort[0]["records"]
    records = manufactured_records(rows)
    candidates = [{"left": "key-0", "right": "key-1"}]
    decisions = [
        {"left": "key-0", "right": "key-1", "decision": "relevant"},
        {"left": "key-2", "right": "key-3", "decision": "irrelevant"},
    ]
    assert relevant_decisions(["key-0"], records, candidates, decisions) == decisions[:1]
    with pytest.raises(ValueError):
        relevant_decisions(["key-0"], records, candidates, [])
    with pytest.raises(ValueError):
        relevant_decisions(["key-0"], records, candidates, decisions[:1] * 2)


@pytest.mark.parametrize(
    "basis", ["collection_batch", "filename", "timestamp", "document_id", "similarity_candidate"]
)
def test_proxies_and_similarity_cannot_document_identity_or_source(basis):
    assert classify_claims([{"value": "manufactured-token", "basis": basis}]) == {
        "value": None,
        "status": "UNVERIFIED",
    }


def test_explicit_relationship_categories_and_conflict_do_not_fabricate_keys():
    explicit = {
        "value": "supplied-recording",
        "basis": "explicit_supplied_mapping",
        "evidence": [{"path": "retained-mapping.json", "sha256": "a" * 64}],
    }
    assert classify_claims([explicit]) == {"value": "supplied-recording", "status": "documented"}
    provisional = {**explicit, "basis": "retained_relationship_hypothesis"}
    assert classify_claims([provisional]) == {"value": None, "status": "provisional"}
    assert classify_claims([explicit, {**explicit, "value": "other-recording"}]) == {
        "value": None,
        "status": "conflicting",
    }
    with pytest.raises(ValueError):
        classify_claims([{**explicit, "evidence": []}])


def test_matrix_keeps_unknown_relationships_despite_lineage_timestamps_and_scene_clue(cohort):
    rows = cohort[0]["records"]
    records = manufactured_records(rows)
    keys = list(records)
    samples = {
        k: {
            "xmp_references": {"documentid": ["shared-template-token"]},
            "exif_datetime_original": "manufactured-time",
            "subject_id": None,
            "source_video_id": None,
        }
        for k in keys
    }
    source = {"samples": samples, "xml_failures": []}
    decisions = [
        {
            "left": keys[0],
            "right": keys[1],
            "review_id": "synthetic-pair",
            "decision": "plausible_shared_scene_distinct_frames",
            "evidence": "uncorroborated scene clue",
        }
    ]
    evidence = [{"path": "synthetic-evidence.json", "sha256": "a" * 64}]
    matrix = evidence_matrix(
        rows, keys, records, source, copy.deepcopy(source), decisions, evidence
    )
    assert all(
        r["value"] is None and r["status"] == "UNVERIFIED"
        for row in matrix
        for r in row["relationships"].values()
    )
    assert matrix[0]["prior_candidate_cautions"][0]["assigned_group"] is None
    assert matrix[0]["prior_candidate_cautions"][0]["relationship_clue_status"] == "provisional"
    summary = aggregate_review(matrix, decisions, source)
    assert summary["readiness_decision"] == "grouped validation unresolved; descriptive work only"
    assert all(summary["relationship_status_counts"][f]["UNVERIFIED"] == 32 for f in RELATIONSHIPS)
    assert summary["within_cohort_candidate_pairs"] == 1
    assert summary["group_keys_created"] == summary["splits_created"] == 0
    assert json.loads(json.dumps(matrix))[0]["relationships"]["subject_id"]["value"] is None
    structured = copy.deepcopy(source)
    structured["samples"][keys[0]]["exif_datetime_original"] = "different"
    with pytest.raises(ValueError, match="disagree"):
        evidence_matrix(rows, keys, records, source, structured, decisions, evidence)


@pytest.mark.parametrize(
    "name",
    [
        "image.png",
        "video.mp4",
        "final.pt",
        "labels.csv",
        "private-predictions.json",
        "paired-images.json",
        "mean-shape.json",
        "private-cohort-permutation.json",
        "manifests/actual.json",
    ],
)
def test_payload_paths_denied_without_opening(tmp_path, name):
    with pytest.raises(ValueError):
        metadata_path(tmp_path / name, tmp_path)


def test_output_refuses_historical_ancestors_descendants_overwrite_and_escape(tmp_path):
    historical = tmp_path / "runs/original"
    historical.mkdir(parents=True)
    for path in (historical, historical / "nested", tmp_path / "runs", tmp_path / "public"):
        with pytest.raises(ValueError):
            new_review_directory(path, tmp_path)
    escape = tmp_path / "runs/escape"
    escape.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        new_review_directory(escape / "new", tmp_path)
    new = new_review_directory(tmp_path / "runs/new-review", tmp_path)
    assert new.is_dir()
    with pytest.raises(ValueError):
        new_review_directory(new, tmp_path)


def test_helper_import_has_no_image_or_model_side_effects():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import infant_movement.evaluation.provenance_readiness; "
            "assert not any(k == 'torch' or k.startswith('PIL') "
            "for k in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0 and result.stderr == ""


def test_local_lock_include_preserves_exact_pins_without_network(tmp_path):
    (tmp_path / "base.lock").write_text("pytest==1.0 --hash=sha256:" + "a" * 64 + "\n")
    lock = tmp_path / "image.lock"
    lock.write_text("-r base.lock\nPillow==2.0 --hash=sha256:" + "b" * 64 + "\n")
    assert locked_pins(lock, tmp_path) == {"pytest": "1.0", "Pillow": "2.0"}


@pytest.mark.parametrize("violation", ["cycle", "escape", "conflict", "unhashed"])
def test_changed_or_unsafe_lock_includes_rejected(tmp_path, violation):
    lock = tmp_path / "image.lock"
    if violation == "cycle":
        text = "-r image.lock\n"
    elif violation == "escape":
        text = "-r ../outside.lock\n"
    elif violation == "conflict":
        text = "pytest==1 --hash=sha256:" + "a" * 64 + "\npytest==2 --hash=sha256:" + "b" * 64
    else:
        text = "pytest>=1\n"
    lock.write_text(text)
    with pytest.raises(ValueError):
        locked_pins(lock, tmp_path)
