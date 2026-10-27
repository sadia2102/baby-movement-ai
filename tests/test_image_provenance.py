import pytest

from infant_movement.evaluation.image_provenance import ProvenanceError, extract_xmp_references


def test_extracts_document_references_and_source_text_without_resolving_urls():
    xml = b"""<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:mm="http://ns.adobe.com/xap/1.0/mm/" xmlns:dc="http://purl.org/dc/elements/1.1/">
    <rdf:RDF><rdf:Description mm:DocumentID="xmp.did:synthetic">
    <dc:source><rdf:Seq><rdf:li>https://example.invalid/not-fetched</rdf:li></rdf:Seq></dc:source>
    <dc:creator>not an infant identity field</dc:creator></rdf:Description></rdf:RDF></x:xmpmeta>"""
    refs = extract_xmp_references(xml)
    assert refs["{http://ns.adobe.com/xap/1.0/mm/}DocumentID"] == ["xmp.did:synthetic"]
    assert refs["{http://purl.org/dc/elements/1.1/}source"] == [
        "https://example.invalid/not-fetched"
    ]
    assert all("creator" not in key for key in refs)


@pytest.mark.parametrize(
    "xml",
    [
        b'<!DOCTYPE x [<!ENTITY e SYSTEM "file:///not-read">]><x>&e;</x>',
        '<!DOCTYPE x [<!ENTITY e "expansion">]><x>&e;</x>'.encode("utf-16"),
        b"<not-closed>",
        b"x" * (1024 * 1024 + 1),
    ],
)
def test_unsafe_or_malformed_xmp_fails_without_echoing_content(xml):
    with pytest.raises(ProvenanceError) as result:
        extract_xmp_references(xml)
    assert "file:///" not in str(result.value)


def test_unrecognized_namespaces_and_unqualified_ids_are_not_source_evidence():
    assert (
        extract_xmp_references(
            '<x xmlns:other="https://example.invalid/" '
            'other:DocumentID="not-evidence" DocumentID="also-not-evidence"/>'
        )
        == {}
    )
