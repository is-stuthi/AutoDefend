"""Shared column-name and label constants for the CAN dataset.

Centralizing these avoids magic strings scattered across the data,
feature, and agent modules.
"""

from __future__ import annotations

CAN_ID_COLUMN = "CAN ID"
LABEL_COLUMN = "Label"
DATA_COLUMNS = [f"DATA[{i}]" for i in range(8)]
REQUIRED_COLUMNS = [CAN_ID_COLUMN, *DATA_COLUMNS, LABEL_COLUMN]

NORMAL_LABEL = "R"

# Maps lowercased raw label values to their canonical display form. The
# source CSV mixes casing (e.g. "gear" vs "DoS"), so labels are normalized
# on load to keep downstream grouping/reporting consistent.
CANONICAL_LABELS: dict[str, str] = {
    "r": "R",
    "dos": "DoS",
    "fuzzy": "Fuzzy",
    "rpm": "RPM",
    "gear": "Gear",
}
