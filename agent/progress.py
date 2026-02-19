import re
from dataclasses import dataclass


@dataclass
class ProgressStats:
    total: int
    done: int

    @property
    def pct(self) -> float:
        if self.total == 0:
            return 0.0
        return round((self.done / self.total) * 100, 2)


def _count_checkboxes(text: str) -> ProgressStats:
    done = len(re.findall(r"- \[x\]", text, flags=re.IGNORECASE))
    total = len(re.findall(r"- \[[x ]\]", text, flags=re.IGNORECASE))
    return ProgressStats(total=total, done=done)


def _section_chunk(content: str, marker: str) -> str:
    start = content.find(marker)
    if start < 0:
        return ""
    next_pos = content.find("\n## ", start + len(marker))
    if next_pos < 0:
        return content[start:]
    return content[start:next_pos]


def summarize_progress(md_path: str = "UPGRADE_PLAN_30J.md") -> str:
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as exc:
        return f"Progress unavailable: {exc}"

    sections = {
        "Semaine 1": "## Semaine 1",
        "Semaine 2": "## Semaine 2",
        "Semaine 3": "## Semaine 3",
        "Semaine 4": "## Semaine 4",
        "KPI": "## KPI cibles a 30 jours",
    }

    stats = {}
    for label, marker in sections.items():
        chunk = _section_chunk(content, marker)
        if not chunk:
            stats[label] = ProgressStats(total=0, done=0)
            continue
        stats[label] = _count_checkboxes(chunk)

    core_content = content
    backlog_marker = "## Backlog bonus"
    backlog_pos = content.find(backlog_marker)
    if backlog_pos >= 0:
        core_content = content[:backlog_pos]
    global_stats = _count_checkboxes(core_content)

    lines = [
        "Progress Summary",
        f"- Global: {global_stats.done}/{global_stats.total} ({global_stats.pct}%)",
        f"- Semaine 1: {stats['Semaine 1'].done}/{stats['Semaine 1'].total} ({stats['Semaine 1'].pct}%)",
        f"- Semaine 2: {stats['Semaine 2'].done}/{stats['Semaine 2'].total} ({stats['Semaine 2'].pct}%)",
        f"- Semaine 3: {stats['Semaine 3'].done}/{stats['Semaine 3'].total} ({stats['Semaine 3'].pct}%)",
        f"- Semaine 4: {stats['Semaine 4'].done}/{stats['Semaine 4'].total} ({stats['Semaine 4'].pct}%)",
        f"- KPI: {stats['KPI'].done}/{stats['KPI'].total} ({stats['KPI'].pct}%)",
    ]

    return "\n".join(lines)
