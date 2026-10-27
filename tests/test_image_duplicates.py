"""Synthetic whole-image audit tests; no infant images, embeddings or network."""

import io
import json
from zipfile import ZipFile

import pytest

from infant_movement.evaluation.image_duplicates import (
    ImageAuditError,
    candidate_pairs,
    exact_groups,
    fingerprint_image,
)
from infant_movement.evaluation.infanface_duplicates import (
    file_sha256,
    private_output_path,
    save_private_audit,
    verify_archive,
)

Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")
ImageOps = pytest.importorskip("PIL.ImageOps")
PngImagePlugin = pytest.importorskip("PIL.PngImagePlugin")


def scene():
    image = Image.new("RGB", (160, 120), (170, 220, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 80, 160, 120), fill=(20, 110, 30))
    draw.rectangle((18, 35, 75, 95), fill=(150, 60, 25))
    draw.polygon(((10, 35), (45, 5), (83, 35)), fill=(240, 30, 20))
    draw.ellipse((110, 5, 145, 40), fill=(250, 210, 10))
    draw.line((0, 119, 160, 75), fill=(30, 30, 30), width=4)
    return image


def encode(image, fmt="PNG", **kwargs):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def test_pixel_equality_ignores_file_metadata_and_lossless_encoding():
    image = scene()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("comment", "synthetic metadata only")
    a = encode(image)
    b = encode(image, pnginfo=metadata, compress_level=1)
    assert a != b
    fa, fb = fingerprint_image(a), fingerprint_image(b)
    assert fa.display_sha256 == fb.display_sha256
    assert exact_groups({"b": fb, "a": fa}) == [["a", "b"]]


def test_shape_is_part_of_pixel_equality():
    a = fingerprint_image(encode(Image.new("RGB", (2, 4), "red")))
    b = fingerprint_image(encode(Image.new("RGB", (4, 2), "red")))
    assert a.display_sha256 != b.display_sha256


def test_exif_orientation_is_recorded_and_display_normalized():
    image = scene()
    rotated = image.transpose(Image.Transpose.ROTATE_90)
    exif = Image.Exif()
    exif[274] = 6
    a = fingerprint_image(encode(image))
    b = fingerprint_image(encode(rotated, exif=exif))
    assert b.orientation == 6
    assert a.stored_sha256 != b.stored_sha256
    assert a.display_sha256 == b.display_sha256
    assert b.display_size == image.size


@pytest.mark.parametrize("transform", ["resize", "jpeg", "mirror", "crop"])
def test_known_image_variants_are_candidates_not_confirmations(transform):
    image = scene()
    altered = image
    fmt, options = "PNG", {}
    if transform == "resize":
        altered = image.resize((320, 240), Image.Resampling.LANCZOS)
    elif transform == "jpeg":
        fmt, options = "JPEG", {"quality": 75}
    elif transform == "mirror":
        altered = ImageOps.mirror(image)
    elif transform == "crop":
        altered = image.crop((16, 12, 144, 108))
    a, b = fingerprint_image(encode(image)), fingerprint_image(encode(altered, fmt, **options))
    pairs = candidate_pairs({"a": a, "b": b})
    assert len(pairs) == 1
    assert pairs[0]["review_status"] == "unreviewed"
    assert pairs[0]["identity_inferred"] is False
    assert a.display_sha256 != b.display_sha256


def test_uniform_hash_collision_is_not_pixel_equality_or_identity():
    a = fingerprint_image(encode(Image.new("RGB", (80, 80), "red")))
    b = fingerprint_image(encode(Image.new("RGB", (80, 80), "blue")))
    pairs = candidate_pairs({"a": a, "b": b})
    assert len(pairs) == 1  # Deliberately documents a false candidate.
    assert pairs[0]["exact_display_pixels"] is False
    assert pairs[0]["rgb_thumbnail_mae"] > 0.5


def test_deterministic_pair_order_and_fixed_thresholds():
    from dataclasses import replace

    image = fingerprint_image(encode(scene()))
    base = replace(image.views[0], dhash=0, ahash=0)
    a = replace(image, display_sha256="a", views=(base,))
    b = replace(image, display_sha256="b", views=(replace(base, dhash=63, ahash=255),))
    c = replace(image, display_sha256="c", views=(replace(base, dhash=127, ahash=255),))
    assert len(candidate_pairs({"a": a, "b": b})) == 1
    assert candidate_pairs({"a": a, "c": c}) == []
    assert candidate_pairs({"b": b, "a": a}) == candidate_pairs({"a": a, "b": b})


@pytest.mark.parametrize("content", [b"not an image", b"\x89PNG\r\n\x1a\n", b"\xff\xd8truncated"])
def test_corrupt_images_fail(content):
    with pytest.raises(ImageAuditError):
        fingerprint_image(content)


def test_multi_frame_and_high_depth_fail():
    frames = [scene(), ImageOps.mirror(scene())]
    animated = encode(frames[0], save_all=True, append_images=frames[1:], duration=100)
    with pytest.raises(ImageAuditError, match="multi-frame"):
        fingerprint_image(animated)
    with pytest.raises(ImageAuditError):
        fingerprint_image(encode(Image.new("I;16", (10, 10))))


def test_dimension_limit_before_pixel_load(monkeypatch):
    import infant_movement.evaluation.image_duplicates as audit

    monkeypatch.setattr(audit, "MAX_PIXELS", 100)
    with pytest.raises(ImageAuditError, match="dimensions"):
        fingerprint_image(encode(scene()))


def test_checksum_mismatch_before_decode(tmp_path, monkeypatch):
    from infant_movement.evaluation.infanface_duplicates import audit_infanface_duplicates

    path = tmp_path / "unchanged.bin"
    path.write_bytes(b"raw synthetic source")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"sha256": "0" * 64}))
    monkeypatch.setattr(
        "infant_movement.evaluation.infanface_duplicates.load_infanface_archive",
        lambda _: pytest.fail("must not parse after failed integrity check"),
    )
    with pytest.raises(ImageAuditError, match="checksum"):
        audit_infanface_duplicates(path, receipt)
    assert path.read_bytes() == b"raw synthetic source"
    assert verify_archive(path, file_sha256(path)) == file_sha256(path)


@pytest.mark.parametrize("digest", [None, "bad", "a" * 63, "A" * 64])
def test_invalid_receipt_digests_fail(tmp_path, digest):
    with pytest.raises(ImageAuditError, match="receipt"):
        verify_archive(tmp_path / "not-opened", digest)


def test_private_output_and_no_overwrite(tmp_path):
    private = tmp_path / "reports/private/result.json"
    assert private_output_path(private, tmp_path) == private
    save_private_audit({"sample": "synthetic-private"}, private, tmp_path)
    with pytest.raises(ImageAuditError, match="overwrite"):
        save_private_audit({}, private, tmp_path)
    with pytest.raises(ImageAuditError, match="private"):
        save_private_audit({}, tmp_path / "reports/public.json", tmp_path)


def test_private_output_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/private").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ImageAuditError, match="private"):
        private_output_path(tmp_path / "reports/private/result.json", tmp_path)


def test_full_synthetic_archive_audit_keeps_unknowns_and_raw_unchanged(
    tmp_path, monkeypatch, capsys
):
    from infant_movement.evaluation.infanface_duplicates import audit_infanface_duplicates

    columns = [
        "image-set",
        "filename",
        "partition",
        "subpartition",
        "turned",
        "occluded",
        "expressive",
        "tilted",
    ]
    columns += [f"gt-{axis}{i}" for i in range(68) for axis in ("x", "y")]
    lines = [",".join(columns)]
    for batch, filename, split, subset in (
        ("ads", "synthetic-a.png", "infanface-train", "infanface-train"),
        ("google", "synthetic-b.png", "infanface-test", "infanface-common"),
    ):
        lines.append(",".join([batch, filename, split, subset, *(["0.0"] * 4), *(["10.5"] * 136)]))
    path = tmp_path / "synthetic.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("infanface-public/labels.csv", "\n".join(lines))
        archive.writestr("infanface-public/images/ads/synthetic-a.png", encode(scene()))
        archive.writestr(
            "infanface-public/images/google/synthetic-b.png", encode(scene(), compress_level=1)
        )
    digest = file_sha256(path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"sha256": digest}))
    report = audit_infanface_duplicates(path, receipt)
    assert report["summary"]["decoded_count"] == 2
    assert report["summary"]["exact_display_pixel_groups"] == 1
    assert report["summary"]["cross_official_partition_candidates"] == 1
    assert all(
        r["subject_id"] is r["source_video_id"] is r["project_split"] is None
        for r in report["records"].values()
    )
    assert file_sha256(path) == digest
    from infant_movement.cli import main

    monkeypatch.chdir(tmp_path)
    output = tmp_path / "reports/private/synthetic-audit.json"
    assert (
        main(
            [
                "duplicate-audit",
                "--archive",
                str(path),
                "--receipt",
                str(receipt),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    assert json.loads(stdout)["decoded_count"] == 2
    assert "synthetic-a.png" not in stdout and "sample-0001" not in stdout
    assert output.is_file()

    import subprocess
    import sys
    from pathlib import Path

    source_output = tmp_path / "reports/private/source-audit.json"
    script = Path(__file__).resolve().parents[1] / "scripts/audit_infanface_sources.py"
    command = [
        sys.executable,
        str(script),
        "--archive",
        str(path),
        "--audit-report",
        str(output),
        "--output",
        str(source_output),
    ]
    process = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    source_summary = json.loads(process.stdout)
    assert source_summary["sample_count"] == 2
    assert source_summary["cross_field_lineage_candidate_pairs"] == 0
    assert source_summary["project_partitions_created"] is False
    assert "synthetic-a.png" not in process.stdout
    assert file_sha256(path) == digest


def test_cli_rejects_public_output_before_opening_input(tmp_path, monkeypatch, capsys):
    from infant_movement.cli import main

    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "duplicate-audit",
                "--archive",
                "not-opened",
                "--receipt",
                "not-opened",
                "--output",
                "reports/public.json",
            ]
        )
        == 2
    )
    assert "private" in json.loads(capsys.readouterr().out)["error"]
    assert not (tmp_path / "reports/public.json").exists()
