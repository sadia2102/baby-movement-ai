"""Conservative extraction of local XMP source/document references, not identity.

Values may describe editing lineage or reused templates, never proven patients
or source videos. No network or external XML entities are supported.
"""

import xml.etree.ElementTree as ET

MAX_XMP_BYTES = 1024 * 1024
FIELDS = {
    "http://ns.adobe.com/xap/1.0/mm/": {"DocumentID", "OriginalDocumentID", "InstanceID"},
    "http://ns.adobe.com/xap/1.0/sType/ResourceRef#": {
        "documentID",
        "originalDocumentID",
        "instanceID",
    },
    "http://purl.org/dc/elements/1.1/": {"source", "relation", "identifier"},
    "http://ns.adobe.com/photoshop/1.0/": {"Source"},
}


class ProvenanceError(ValueError):
    """XMP cannot be read under the bounded, entity-free local protocol."""


def _allowed(key):
    if not key.startswith("{") or "}" not in key:
        return False
    namespace, local = key[1:].split("}", 1)
    return local in FIELDS.get(namespace, set())


def extract_xmp_references(content: bytes | str) -> dict[str, list[str]]:
    """Extract whitelisted source/lineage text; never resolve URLs or identify people."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    if len(content) > MAX_XMP_BYTES:
        raise ProvenanceError("XMP exceeds audit limit")
    # Reject declarations, including UTF-16/32 encoded variants, before parsing.
    declarations = content.replace(b"\0", b"").upper()
    if b"<!DOCTYPE" in declarations or b"<!ENTITY" in declarations:
        raise ProvenanceError("XMP entity/DOCTYPE declarations are unsupported")
    try:
        root = ET.fromstring(content)
    except ET.ParseError as error:
        raise ProvenanceError("malformed XMP") from error
    result = {}

    def add(key, text):
        if _allowed(key) and text and text.strip():
            value = text.strip()
            if len(value) > 4096:
                raise ProvenanceError("XMP reference exceeds audit limit")
            result.setdefault(key, set()).add(value)

    for node in root.iter():
        for key, value in node.attrib.items():
            add(key, value)
        if _allowed(node.tag):
            add(node.tag, node.text)
            for child in node.iter():
                if child.tag == "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li":
                    add(node.tag, child.text)
    return {key: sorted(values) for key, values in sorted(result.items())}
