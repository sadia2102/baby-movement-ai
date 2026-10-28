import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from infant_movement.datasets.infanface import COLUMNS


@pytest.fixture
def document():
    return json.loads((Path(__file__).parent / "fixtures/synthetic_metadata.json").read_text())


@pytest.fixture
def pilot_archive():
    Image = pytest.importorskip("PIL.Image")

    def manufacture(tmp_path, *, degenerate=False, orientation=0):
        """Two fake train images; test numeric payload is deliberately unparsable."""
        image = Image.new("RGB", (20, 12), (150, 50, 90))
        buffer = io.BytesIO()
        exif = Image.Exif()
        exif[274] = orientation
        image.save(buffer, format="PNG", exif=exif)
        image.close()
        text = io.StringIO()
        writer = csv.DictWriter(text, fieldnames=COLUMNS)
        writer.writeheader()
        test = dict.fromkeys(COLUMNS, "MUST-NOT-NUMERICALLY-PARSE")
        test.update(partition="infanface-test")
        writer.writerow(test)
        for name in ("manufactured-a.png", "manufactured-b.png"):
            row = dict.fromkeys(COLUMNS, "0.0")
            row.update(
                {
                    "image-set": "ads",
                    "filename": name,
                    "partition": "infanface-train",
                    "subpartition": "infanface-train",
                }
            )
            for i in range(68):
                x, y = ((4, 3), (14, 3), (14, 8), (4, 8))[i % 4]
                if degenerate and name.endswith("b.png"):
                    x, y = 4, 3
                row[f"gt-x{i}"], row[f"gt-y{i}"] = str(x), str(y)
            if not degenerate:
                row["gt-x67"] = "-10"
            writer.writerow(row)
        path = tmp_path / "manufactured.zip"
        with ZipFile(path, "w") as archive:
            archive.writestr("infanface-public/labels.csv", text.getvalue())
            for name in ("manufactured-a.png", "manufactured-b.png"):
                archive.writestr(f"infanface-public/images/ads/{name}", buffer.getvalue())
            archive.writestr("infanface-public/images/google/forbidden.png", b"never open")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    return manufacture
