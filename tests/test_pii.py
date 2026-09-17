import pytest
from audit import EnterprisePIIEngine

@pytest.fixture
def pii_engine():
    return EnterprisePIIEngine()

def test_email_redaction(pii_engine):
    text = "Please reach out to alice.smith@internal-corp.com regarding access."
    anonymized, record = pii_engine.scan_and_anonymize(text, source_doc="test_doc")
    assert "alice.smith@internal-corp.com" not in anonymized
    assert "<EMAIL_ADDRESS_1>" in anonymized
    assert record.detected_entities_count == 1
    assert record.entities[0].entity_type == "EMAIL_ADDRESS"

def test_ip_address_redaction(pii_engine):
    text = "The gateway is running on 192.168.1.105 with failover at 10.0.0.1."
    anonymized, record = pii_engine.scan_and_anonymize(text, source_doc="test_doc")
    assert "192.168.1.105" not in anonymized
    assert "10.0.0.1" not in anonymized
    assert record.detected_entities_count >= 2
    types = [e.entity_type for e in record.entities]
    assert "IP_ADDRESS" in types

def test_phone_number_redaction(pii_engine):
    text = "Direct hotline: +1-555-839-2041 or 555-123-4567 for urgent incidents."
    anonymized, record = pii_engine.scan_and_anonymize(text, source_doc="test_doc")
    assert "555-839-2041" not in anonymized
    assert record.detected_entities_count >= 1
    types = [e.entity_type for e in record.entities]
    assert "PHONE_NUMBER" in types

def test_person_redaction(pii_engine):
    text = "Architect Dr. Sanjay reviewed the technical proposal with Ms. Sarah Connor."
    anonymized, record = pii_engine.scan_and_anonymize(text, source_doc="test_doc")
    assert "Dr. Sanjay" not in anonymized
    assert "Ms. Sarah Connor" not in anonymized
    assert record.detected_entities_count >= 2
    types = [e.entity_type for e in record.entities]
    assert "PERSON" in types

def test_no_pii_passthrough(pii_engine):
    clean_text = "The database cluster consists of 3 read replicas with raft consensus protocol."
    anonymized, record = pii_engine.scan_and_anonymize(clean_text, source_doc="test_doc")
    assert anonymized == clean_text
    assert record.detected_entities_count == 0
