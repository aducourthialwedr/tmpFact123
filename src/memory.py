"""Suivi de la consommation mémoire pendant l'exécution (notebook, pod à mémoire limitée).

    from src.memory import MemoryMonitor
    monitor = MemoryMonitor("reports/memory.csv").start()
    project = Project(dataset="real", log=monitor.log)     # chaque message suivi de la mémoire
    ...
    monitor.by_phase()                                    # pic par étape / sous-étape

Un fil d'arrière-plan relève chaque seconde :
- la mémoire résidente du processus (RSS) et son pic ;
- la mémoire du conteneur (cgroup v1 ou v2) : consommée, anonyme (ce que l'OOM killer compte
  réellement, hors cache disque) et limite du pod ;
- la phase en cours : étape de `Project`, jour du rejeu, sous-étape (signal d'allocation, règle,
  candidats ML...), posée par `mark_step` / `mark_day` / `mark` depuis le code de la pipeline.

Chaque relevé est **écrit et vidé immédiatement** dans un CSV : si le noyau est tué (OOM), le fichier
reste, et `by_phase(path)` / `chart(path)` montrent, dans un nouveau noyau, la phase où la mémoire a
explosé. `tail -f` sur ce fichier depuis un terminal JupyterLab donne un suivi en direct.

`mark*` ne coûtent rien sans moniteur actif. Aucun appel réseau.
"""

from __future__ import annotations

import csv
import gc
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

GB = 2 ** 30
COLUMNS = ["time", "elapsed_s", "rss_gb", "rss_peak_gb", "pod_used_gb", "pod_anon_gb", "pod_limit_gb",
           "step", "day", "sub"]

_START = "<démarrage du suivi>"                  # ligne séparant les sessions dans le CSV

_ACTIVE: MemoryMonitor | None = None


# --- Relevés -------------------------------------------------------------------------------------------

def process_rss() -> int:
    """Mémoire résidente du processus, en octets (psutil si présent, sinon /proc)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        pass
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _read_int(path: str) -> int | None:
    try:
        text = Path(path).read_text().strip()
    except OSError:
        return None
    return None if not text or text == "max" else int(text)


def _stat(path: str, key: str) -> int | None:
    try:
        for line in Path(path).read_text().splitlines():
            name, _, value = line.partition(" ")
            if name == key:
                return int(value)
    except OSError:
        pass
    return None


def pod_memory() -> dict[str, int | None]:
    """Mémoire du conteneur (cgroup v2, sinon v1) : consommée, anonyme, limite. None hors conteneur."""
    if Path("/sys/fs/cgroup/memory.current").exists():                      # cgroup v2
        return {"used": _read_int("/sys/fs/cgroup/memory.current"),
                "anon": _stat("/sys/fs/cgroup/memory.stat", "anon"),
                "limit": _read_int("/sys/fs/cgroup/memory.max")}
    base = "/sys/fs/cgroup/memory/"                                          # cgroup v1
    if Path(base + "memory.usage_in_bytes").exists():
        limit = _read_int(base + "memory.limit_in_bytes")
        return {"used": _read_int(base + "memory.usage_in_bytes"),
                "anon": _stat(base + "memory.stat", "total_rss"),
                "limit": limit if limit and limit < 2 ** 60 else None}      # « illimité » = très grand nombre
    return {"used": None, "anon": None, "limit": None}


def release_memory() -> None:
    """Rend au système la mémoire libérée : ramasse-miettes, puis `malloc_trim` (glibc, Linux).

    Sans `malloc_trim`, la mémoire libérée par pandas / numpy reste souvent réservée au processus : le
    RSS ne redescend pas d'un jour ou d'une étape à l'autre et le pod finit par être tué.
    """
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


# --- Phases (appelées depuis la pipeline) ----------------------------------------------------------------

def mark(sub: str) -> None:
    """Sous-étape en cours (signal d'allocation, règle, candidats ML...)."""
    if _ACTIVE is not None:
        _ACTIVE.sub = sub


def mark_step(step: str) -> None:
    """Étape en cours (méthode de `Project`) ; remet à zéro jour et sous-étape."""
    if _ACTIVE is not None:
        _ACTIVE.step, _ACTIVE.day, _ACTIVE.sub = step, "", ""


def mark_day(day, batch: int) -> None:
    """Début d'un jour du rejeu (appelé par `run_replay`)."""
    if _ACTIVE is not None:
        _ACTIVE.day, _ACTIVE.sub = f"{pd.Timestamp(day).date()} (lot {batch})", ""


def end_day() -> None:
    """Fin d'un jour du rejeu : mémoire rendue au système si le moniteur le demande."""
    if _ACTIVE is not None and _ACTIVE.trim_daily:
        release_memory()


# --- Moniteur ---------------------------------------------------------------------------------------------

def _gb(value: int | None) -> float | None:
    return None if value is None else round(value / GB, 3)


class MemoryMonitor:
    """Relevé périodique de la mémoire dans un CSV (vidé à chaque ligne) et suivi des phases.

    `trim_daily` : rend la mémoire libérée au système à la fin de chaque jour du rejeu.
    `warn_ratio` : part de la limite du pod au-delà de laquelle les messages portent une alerte.
    """

    def __init__(self, path: str | Path = "reports/memory.csv", interval: float = 1.0, trim_daily: bool = True,
                 warn_ratio: float = 0.85, echo: Callable[[str], None] = print):
        self.path = Path(path)
        self.interval = interval
        self.trim_daily = trim_daily
        self.warn_ratio = warn_ratio
        self.echo = echo
        self.step = self.day = self.sub = ""
        self.peak = 0
        self.last: dict = {}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._file = None
        self._t0 = time.time()
        self._live = None

    # Démarrage / arrêt ------------------------------------------------------------------------------------

    def start(self) -> MemoryMonitor:
        global _ACTIVE
        if _ACTIVE is not None and _ACTIVE is not self:
            _ACTIVE.stop()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not self.path.exists() or self.path.stat().st_size == 0
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if fresh:
            self._writer.writerow(COLUMNS)
        self._writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), *[""] * 6, _START, "", ""])
        self._t0 = time.time()
        self._stop.clear()
        _ACTIVE = self
        self.sample()
        self._thread = threading.Thread(target=self._run, name="memory-monitor", daemon=True)
        self._thread.start()
        limit = self.last.get("pod_limit_gb")
        self.echo(f"[mémoire] suivi → {self.path.resolve()} ; limite du pod : "
                  f"{f'{limit:.1f} Go' if limit else 'non détectée (hors conteneur ?)'}")
        return self

    def stop(self) -> None:
        global _ACTIVE
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        with self._lock:
            if self._file is not None and not self._file.closed:
                self._file.close()
        if _ACTIVE is self:
            _ACTIVE = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sample()
                if self._live is not None:
                    self._live.update(self._badge())
            except Exception:                    # le suivi ne doit jamais interrompre la pipeline
                pass

    def sample(self) -> dict:
        rss = process_rss()
        self.peak = max(self.peak, rss)
        pod = pod_memory()
        row = {"time": time.strftime("%H:%M:%S"), "elapsed_s": round(time.time() - self._t0, 1),
               "rss_gb": _gb(rss), "rss_peak_gb": _gb(self.peak), "pod_used_gb": _gb(pod["used"]),
               "pod_anon_gb": _gb(pod["anon"]), "pod_limit_gb": _gb(pod["limit"]),
               "step": self.step, "day": self.day, "sub": self.sub}
        self.last = row
        with self._lock:
            if self._file is not None and not self._file.closed:
                self._writer.writerow([row[c] for c in COLUMNS])
                self._file.flush()
        return row

    # Affichage ---------------------------------------------------------------------------------------------

    def status(self) -> str:
        r = self.last or self.sample()
        text = f"RSS {r['rss_gb']:.2f} Go (pic {r['rss_peak_gb']:.2f})"
        if r["pod_used_gb"] is not None:
            limit = r["pod_limit_gb"]
            anon = f", anonyme {r['pod_anon_gb']:.2f}" if r["pod_anon_gb"] is not None else ""
            text += f" · pod {r['pod_used_gb']:.2f}{f'/{limit:.1f}' if limit else ''} Go{anon}"
            used = r["pod_anon_gb"] if r["pod_anon_gb"] is not None else r["pod_used_gb"]
            if limit and used >= self.warn_ratio * limit:
                text += " ⚠ PROCHE DE LA LIMITE"
        return text

    def log(self, message: str) -> None:
        """À passer comme `log` de `Project` : chaque message est suivi de l'état mémoire."""
        if message.startswith("… "):
            mark_step(message[2:])
        self.echo(f"{message}   [{self.status()}]")

    def _badge(self):
        from IPython.display import HTML
        phase = " › ".join(p for p in (self.step, self.day, self.sub) if p) or "—"
        return HTML(f"<code>{self.last.get('time', '')} · {self.status()}<br>{phase}</code>")

    def live(self) -> None:
        """Pastille mise à jour chaque seconde dans la sortie de la cellule qui l'appelle (Jupyter)."""
        from IPython.display import display
        self._live = display(self._badge(), display_id=True)

    # Analyse -----------------------------------------------------------------------------------------------

    def history(self) -> pd.DataFrame:
        return read_history(self.path)

    def by_phase(self) -> pd.DataFrame:
        return by_phase(self.path)

    def chart(self):
        return chart(self.path)


# --- Analyse, y compris après un arrêt brutal du noyau -------------------------------------------------------

def read_history(path: str | Path = "reports/memory.csv") -> pd.DataFrame:
    """Relevés du CSV ; une session par démarrage du suivi (ligne « démarrage » écrite par `start`)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["session"] = (df["step"] == _START).cumsum()
    df = df[df["step"] != _START].copy()
    for c in COLUMNS:
        if c.endswith("_gb") or c == "elapsed_s":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def _session(df: pd.DataFrame, session: int | None) -> pd.DataFrame:
    if session is None or df.empty:
        return df
    return df[df["session"] == df["session"].unique()[session]]


def by_phase(path: str | Path = "reports/memory.csv", session: int | None = -1) -> pd.DataFrame:
    """Pic de mémoire par (étape, sous-étape), dans l'ordre d'exécution, pour la dernière session
    (`session=None` : toutes). Mesure : mémoire anonyme du pod si disponible, sinon RSS du processus."""
    df = _session(read_history(path), session)
    mem = "pod_anon_gb" if df["pod_anon_gb"].notna().any() else "rss_gb"
    keys = ["step", "sub"]
    out = (df.groupby(keys, sort=False)
           .agg(début=("time", "first"), fin=("time", "last"), relevés=("time", "size")).reset_index())
    at_peak = df.sort_values(mem, ascending=False, kind="stable").drop_duplicates(keys)[[*keys, mem, "day"]]
    return out.merge(at_peak, on=keys, how="left").rename(
        columns={"step": "étape", "sub": "sous-étape", mem: f"pic_{mem}", "day": "jour_du_pic"})


def chart(path: str | Path = "reports/memory.csv", session: int = -1):
    """Courbe mémoire de la session (altair) ; limite du pod en rouge, phases en infobulle."""
    import altair as alt
    df = _session(read_history(path), session)
    cols = [c for c in ("rss_gb", "pod_anon_gb", "pod_used_gb") if df[c].notna().any()]
    long = df.melt(id_vars=["elapsed_s", "step", "day", "sub"], value_vars=cols, var_name="mesure",
                   value_name="Go")
    layers = [alt.Chart(long).mark_line().encode(
        x=alt.X("elapsed_s:Q", title="secondes"), y=alt.Y("Go:Q"), color="mesure:N",
        tooltip=["elapsed_s", "mesure", "Go", "step", "day", "sub"])]
    limit = df["pod_limit_gb"].dropna()
    if len(limit):
        layers.append(alt.Chart(pd.DataFrame({"y": [limit.iloc[-1]]}))
                      .mark_rule(color="red", strokeDash=[4, 4]).encode(y="y:Q"))
    return alt.layer(*layers).properties(height=260, title="Mémoire (limite du pod en rouge)")
