"""Metadata-only evidence joins; no dataset, image, model or partition operations."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

RELATIONSHIPS = ("subject_id", "source_video_id", "source_group_id", "session_id")
STATUSES = ("documented", "provisional", "conflicting", "UNVERIFIED")
SEALED_NAMES = {
    "private-predictions.json",
    "paired-images.json",
    "mean-shape.json",
    "private-cohort-permutation.json",
    "selection.json.sealed",
}


def metadata_path(path, root):
    root, path = Path(root).resolve(), Path(path).resolve()
    if (
        not path.is_relative_to(root)
        or path.suffix not in (".json", ".md", ".py", ".lock", ".toml", ".txt", ".html")
        or path.name in SEALED_NAMES
        or path.is_relative_to(root / "manifests")
    ):
        raise ValueError("only permitted local metadata/documents may be read")
    return path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def locked_pins(path, lock_root, seen=None):
    """Resolve the retained local -r include; no dependency/network resolution."""
    path, lock_root = Path(path).resolve(), Path(lock_root).resolve()
    seen = set() if seen is None else seen
    if not path.is_relative_to(lock_root) or path.suffix != ".lock" or path in seen:
        raise ValueError("invalid or cyclic local lock include")
    seen.add(path)
    pins = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith("-r "):
            values = locked_pins(path.parent / line[3:], lock_root, seen)
        else:
            match = re.fullmatch(r"([\w.-]+)==([^\s]+) --hash=sha256:[0-9a-f]{64}", line)
            if match is None:
                raise ValueError("unexpected retained lock syntax")
            values = {match[1]: match[2]}
        if any(name in pins and pins[name] != version for name, version in values.items()):
            raise ValueError("conflicting retained dependency pins")
        pins.update(values)
    return pins


def frozen_cohort(selection, prepared):
    rows = selection["records"]
    refs = [
        {k: r[k] for k in ("image_set", "filename", "source_row", "official_partition")}
        for r in rows
    ]
    if (
        len(rows) != 32
        or len({(r["image_set"], r["filename"]) for r in rows}) != 32
        or canonical_hash(refs) != prepared["selection_sha256"]
        or selection["selection_sha256"] != prepared["selection_sha256"]
        or any(
            r["official_partition"] != "infanface-train"
            or r["image_set"] not in ("ads", "google2", "youtube2")
            or r["eligible"] is not True
            or r["valid_points"] != 68
            for r in rows
        )
    ):
        raise ValueError("requires the unchanged ordered original usable training cohort")
    return rows


def match_cohort(rows, records):
    """Inspect membership/reference fields first; never inspect test evidence."""
    wanted = {(r["image_set"], r["filename"]): r for r in rows}
    matches = {}
    for key, record in records.items():
        if record["official_partition"] != "infanface-train":
            continue
        path = Path(record["archive_image_path"])
        ref = (path.parent.name, path.name)
        if ref not in wanted:
            continue
        row = wanted[ref]
        if record["source_row"] != row["source_row"] or ref in matches:
            raise ValueError("ambiguous or changed historical reference join")
        matches[ref] = key
    if set(matches) != set(wanted):
        raise ValueError("missing cohort reference in retained metadata")
    return [matches[(r["image_set"], r["filename"])] for r in rows]


def relevant_decisions(selected, records, candidates, decisions):
    selected, allowed = set(selected), set()
    for pair in candidates:
        endpoints = (pair["left"], pair["right"])
        if not selected.intersection(endpoints):
            continue
        if any(records[k]["official_partition"] != "infanface-train" for k in endpoints):
            continue
        allowed.add(tuple(sorted(endpoints)))
    result, seen = [], set()
    for decision in decisions:
        # Do not match missing optional IDs: only exact endpoint pairs qualify.
        key = tuple(sorted((decision["left"], decision["right"])))
        if key not in allowed:
            continue
        if key in seen:
            raise ValueError("duplicate retained candidate decision")
        seen.add(key)
        result.append(decision)
    if seen != allowed:
        raise ValueError("relevant candidate lacks a retained decision")
    return result


def classify_claims(claims):
    """Only explicit supplied mappings can document a relationship; no inferred IDs."""
    supported = [
        c
        for c in claims
        if c.get("value") is not None
        and c.get("basis") in ("explicit_supplied_mapping", "retained_relationship_hypothesis")
    ]
    if not supported:
        return {"value": None, "status": "UNVERIFIED"}
    if any(not c.get("evidence") for c in supported):
        raise ValueError("relationship claims require traceable evidence")
    if len({c["value"] for c in supported}) != 1:
        return {"value": None, "status": "conflicting"}
    documented = any(c["basis"] == "explicit_supplied_mapping" for c in supported)
    return {
        "value": supported[0]["value"] if documented else None,
        "status": "documented" if documented else "provisional",
    }


def evidence_matrix(rows, keys, records, source, structured, decisions, evidence):
    matrix = []
    for position, (row, key) in enumerate(zip(rows, keys, strict=True)):
        local = source["samples"][key]
        if local != structured["samples"][key]:
            raise ValueError("retained local-source extractions disagree")
        # These historical records have no authoritative relationship mappings.
        # Unexpected nonnull IDs require human review; do not promote an ID alone.
        observations = [row, records[key], local]
        relationships = {}
        for field in RELATIONSHIPS:
            values = {r[field] for r in observations if r.get(field) is not None}
            relationships[field] = {
                "value": None,
                "status": "conflicting" if len(values) > 1 else "UNVERIFIED",
                "unsupported_observed_values": sorted(values),
                "evidence": evidence,
                "missing": "explicit supplied relationship mapping and overlap resolution",
            }
        cautions = []
        for d in decisions:
            if key not in (d["left"], d["right"]):
                continue
            if d.get("source_video_confirmed") or any(
                d.get(field) is not None for field in RELATIONSHIPS
            ):
                raise ValueError("a retained mapping requires explicit relationship adjudication")
            if d["decision"] == "plausible_shared_scene_distinct_frames":
                status = "provisional"
            elif d["decision"] == "distinct_scene_shared_letterbox_presentation":
                status = "provisional"
            elif d["decision"] == "layout_only_no_image_duplicate_evidence":
                status = "UNVERIFIED"
            else:
                raise ValueError("unreviewed retained decision category")
            cautions.append(
                {
                    "review_id": d["review_id"],
                    "peer_training_reference_key": d["right"] if key == d["left"] else d["left"],
                    "prior_decision": d["decision"],
                    "relationship_clue_status": status,
                    "evidence_text": d["evidence"],
                    "source_video_confirmed": False,
                    "assigned_group": None,
                }
            )
        matrix.append(
            {
                "cohort_position": position,
                "reference": {
                    k: row[k] for k in ("image_set", "filename", "source_row", "official_partition")
                },
                "collection_batch_status": "documented membership; not an independent group",
                "historical_metadata_key": key,
                "relationships": relationships,
                "evidence_pointers": {
                    "selection_position": position,
                    "duplicate_record_key": key,
                    "source_sample_key": key,
                },
                "retained_xmp_references": local["xmp_references"],
                "retained_exif_datetime_original": local["exif_datetime_original"],
                "metadata_tokens_establish_recording_or_session": False,
                "prior_candidate_cautions": cautions,
                "unresolved_grouping_obstacles": [
                    "no supported subject/recording/source-group/session mapping",
                    "repeat subjects, reuploads and shared recordings remain unexcluded",
                    "batch membership and absence of detected duplicates do not prove independence",
                ],
            }
        )
    return matrix


def aggregate_review(matrix, decisions, source):
    counts = {field: dict.fromkeys(STATUSES, 0) for field in RELATIONSHIPS}
    for row in matrix:
        for field, value in row["relationships"].items():
            counts[field][value["status"]] += 1
    keys = {r["historical_metadata_key"] for r in matrix}
    return {
        "cohort_images": len(matrix),
        "official_training_images": len(matrix),
        "collection_batch_counts": dict(Counter(r["reference"]["image_set"] for r in matrix)),
        "relationship_status_counts": counts,
        "relevant_candidate_pairs": len(decisions),
        "prior_candidate_decision_counts": dict(Counter(d["decision"] for d in decisions)),
        "cohort_images_touching_prior_candidates": sum(
            bool(r["prior_candidate_cautions"]) for r in matrix
        ),
        "within_cohort_candidate_pairs": sum(
            d["left"] in keys and d["right"] in keys for d in decisions
        ),
        "cohort_to_other_training_candidate_pairs": sum(
            not (d["left"] in keys and d["right"] in keys) for d in decisions
        ),
        "cohort_images_with_xmp_references": sum(
            bool(r["retained_xmp_references"]) for r in matrix
        ),
        "cohort_images_with_retained_original_datetime": sum(
            r["retained_exif_datetime_original"] is not None for r in matrix
        ),
        "cohort_xmp_parse_failures": sum(f["sample"] in keys for f in source["xml_failures"]),
        "newly_established_grouping_relationships": 0,
        "subject_disjoint_validation_supported": False,
        "source_disjoint_validation_supported": False,
        "readiness_decision": "grouped validation unresolved; descriptive work only",
        "group_keys_created": 0,
        "splits_created": 0,
    }


def new_review_directory(path, root):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root / "runs" or not path.is_relative_to(root / "runs"):
        raise ValueError("review outputs require a new ignored run directory")
    if (root / "runs").exists():
        for prior in (root / "runs").iterdir():
            if path.is_relative_to(prior) or prior.is_relative_to(path):
                raise ValueError("refuse overwriting or nesting inside historical runs")
    path.mkdir(parents=True, exist_ok=False)
    return path
