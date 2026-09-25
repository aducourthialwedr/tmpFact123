"""Suivi mémoire : relevés écrits au fil de l'eau, phases, analyse après coup."""

from src import memory
from src.memory import MemoryMonitor, by_phase, read_history


def test_monitor_writes_samples_with_phases(tmp_path):
    path = tmp_path / "mem.csv"
    lines = []
    m = MemoryMonitor(path, interval=0.05, echo=lines.append).start()
    try:
        memory.mark_step("allocation validation")
        memory.mark_day("2024-09-01", 1234)
        memory.mark("allocation · référence · bloc 1/1")
        m.sample()
        m.log("  2024-09-01")
    finally:
        m.stop()
    assert memory._ACTIVE is None
    h = read_history(path)
    assert len(h) >= 2 and (h["rss_gb"] > 0).all()
    phases = by_phase(path)
    row = phases[phases["sous-étape"] == "allocation · référence · bloc 1/1"].iloc[0]
    assert row["étape"] == "allocation validation" and row["jour_du_pic"].startswith("2024-09-01")
    assert "RSS" in lines[-1] and lines[-1].startswith("  2024-09-01")


def test_sessions_are_separated(tmp_path):
    path = tmp_path / "mem.csv"
    for step in ("a", "b"):
        m = MemoryMonitor(path, interval=10, echo=lambda _: None).start()
        memory.mark_step(step)
        m.sample()
        m.stop()
    h = read_history(path)
    assert h["session"].nunique() == 2
    steps = set(by_phase(path)["étape"])
    assert "b" in steps and "a" not in steps


def test_marks_without_monitor_are_noops():
    assert memory._ACTIVE is None
    memory.mark("x")
    memory.mark_day("2024-01-01", 1)
    memory.end_day()
