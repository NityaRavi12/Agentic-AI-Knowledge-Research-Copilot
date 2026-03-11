"""
Extract technique documents from MITRE ATT&CK Enterprise STIX bundle.
Outputs docs.jsonl with one record per technique (description section for MVP).
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

# Technique ID pattern: T followed by digits, optional .digits for sub-technique
TECHNIQUE_ID_RE = re.compile(r"^T\d+(?:\.\d+)?$")


def load_stix(path: str | Path) -> dict[str, Any]:
    """Load STIX 2.1 bundle from JSON file. Returns the full bundle dict."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _technique_id_from_refs(external_references: list[dict]) -> str | None:
    """Extract technique ID (e.g. T1059, T1059.001) from external_references."""
    for ref in external_references or []:
        eid = ref.get("external_id")
        if eid and TECHNIQUE_ID_RE.match(eid):
            return eid
    return None


def _url_for_technique(technique_id: str) -> str:
    """Build MITRE ATT&CK technique URL. base_tid = part before dot (e.g. T1059 for T1059.001)."""
    base_tid = technique_id.split(".")[0] if "." in technique_id else technique_id
    return f"https://attack.mitre.org/techniques/{base_tid}/"


def extract_technique_docs(stix: dict[str, Any]) -> list[dict]:
    """
    Filter attack-pattern objects and convert to doc records.
    Uses name as title, description as text; section is "description" for MVP.
    """
    objects = stix.get("objects") or []
    records = []
    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        technique_id = _technique_id_from_refs(obj.get("external_references") or [])
        if not technique_id:
            continue
        name = obj.get("name") or ""
        description = obj.get("description") or ""
        text = " ".join(description.split())
        section = "description"
        doc_id = f"{technique_id}::{section}"
        url = _url_for_technique(technique_id)
        records.append({
            "doc_id": doc_id,
            "technique_id": technique_id,
            "title": name,
            "section": section,
            "url": url,
            "text": text,
        })
    return records


def write_jsonl(records: list[dict], out_path: str | Path) -> None:
    """Write one JSON object per line to out_path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    raw_path = root / "data" / "raw" / "enterprise-attack.json"
    out_path = root / "data" / "processed" / "docs.jsonl"
    if len(sys.argv) >= 2:
        raw_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2])
    stix = load_stix(raw_path)
    records = extract_technique_docs(stix)
    write_jsonl(records, out_path)
    print(f"Wrote {len(records)} docs to {out_path}")


if __name__ == "__main__":
    main()
