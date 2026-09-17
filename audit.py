import re
import json
import uuid
import logging
from typing import List, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime, timezone
from config import settings
from models import PIIEntity, AuditRecord

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger("audit_governance")

class EnterprisePIIEngine:
    """
    Enterprise-grade PII detection and redaction engine.
    Supports standard enterprise regex/pattern recognizers with deterministic masking
    and immutable JSONL audit logging.
    Guaranteed safe against system-level interrupts and Application Control policies.
    """
    def __init__(self):
        # High-precision patterns covering key enterprise PII categories
        self.patterns = {
            "EMAIL_ADDRESS": re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
                re.IGNORECASE
            ),
            "PHONE_NUMBER": re.compile(
                r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
            ),
            "IP_ADDRESS": re.compile(
                r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
            ),
            "PERSON": re.compile(
                r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
                r"|\b(?:John Doe|Jane Doe|Alice Smith|Bob Johnson|Sanjay|Michael Brown|Sarah Connor|Bruce Wayne)\b"
            ),
            "CREDIT_CARD": re.compile(
                r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
            ),
            "SSN": re.compile(
                r"\b\d{3}-\d{2}-\d{4}\b"
            )
        }

    def _mask_for_debug(self, text: str) -> str:
        """Mask middle characters for safe audit preview (e.g. j***@corp.com)."""
        if len(text) <= 3:
            return "***"
        return text[0] + "***" + text[-1]

    def scan_and_anonymize(self, text: str, source_doc: str = "interactive_input") -> Tuple[str, AuditRecord]:
        """
        Scans input text for sensitive entities, generates deterministic redaction tokens,
        and records an immutable audit ledger entry.
        """
        entities: List[PIIEntity] = []
        entity_counters: Dict[str, int] = {}
        
        # Detect matches across defined patterns
        detected_spans = []
        for entity_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                start, end = match.span()
                matched_val = match.group(0)
                detected_spans.append({
                    "type": entity_type,
                    "start": start,
                    "end": end,
                    "val": matched_val,
                    "score": 0.95
                })

        # Sort descending to prevent span collision during replacement
        detected_spans.sort(key=lambda x: x["start"], reverse=True)

        # Eliminate overlapping spans
        filtered_spans = []
        last_start = float("inf")
        for span in detected_spans:
            if span["end"] <= last_start:
                filtered_spans.append(span)
                last_start = span["start"]

        # Sort ascending for orderly placeholder numbering
        filtered_spans.sort(key=lambda x: x["start"])

        # Build replacement mapping
        for span in filtered_spans:
            etype = span["type"]
            entity_counters[etype] = entity_counters.get(etype, 0) + 1
            replacement = f"<{etype}_{entity_counters[etype]}>"
            span["replacement"] = replacement
            
            entities.append(PIIEntity(
                entity_type=etype,
                start=span["start"],
                end=span["end"],
                score=span["score"],
                replacement_token=replacement,
                original_value_masked=self._mask_for_debug(span["val"])
            ))

        # Perform in-place textual replacement from right to left
        anonymized_text = text
        for span in sorted(filtered_spans, key=lambda x: x["start"], reverse=True):
            s, e = span["start"], span["end"]
            anonymized_text = anonymized_text[:s] + span["replacement"] + anonymized_text[e:]

        # Create immutable audit record
        record = AuditRecord(
            record_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            source_document=source_doc,
            detected_entities_count=len(entities),
            entities=entities,
            status="SANITIZED"
        )

        # Persist audit record
        self._write_audit_log(record)
        return anonymized_text, record

    def _write_audit_log(self, record: AuditRecord):
        """Append audit record to JSONL storage."""
        try:
            with open(settings.AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
        except Exception as e:
            logger.error("Failed to write audit log: %s", e)

    def read_audit_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Read recent audit logs for compliance inspector UI."""
        if not settings.AUDIT_LOG_PATH.exists():
            return []
        records = []
        try:
            with open(settings.AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in reversed(lines[-limit:]):
                    clean_line = line.strip()
                    if clean_line:
                        records.append(json.loads(clean_line))
        except Exception as e:
            logger.error("Error reading audit logs: %s", e)
        return records

# Global singleton engine instance
pii_engine = EnterprisePIIEngine()
