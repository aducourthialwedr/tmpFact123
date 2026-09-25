"""Construit le livrable transportable : dist/reconciliation_poc.zip (outil de développement).

Contenu : code, configuration, notebook, dépendances, documentation. Exclus : tests, outils de
développement, données, rapports, modèles, caches.

    python tools/build_bundle.py
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "config", "notebooks", ".streamlit", "pyproject.toml", "requirements.txt", "README.md",
           "GUIDE.md", "CLAUDE.md", "spec_rapprochement_automatique.md"]
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
EMPTY_DIRS = ["data", "reports", "models"]


def files() -> list[Path]:
    out = []
    for entry in INCLUDE:
        path = ROOT / entry
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        out += [p for p in candidates if not EXCLUDED_PARTS & set(p.relative_to(ROOT).parts)]
    return out


def main() -> Path:
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    target = dist / "reconciliation_poc.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files():
            z.write(f, Path("reconciliation_poc") / f.relative_to(ROOT))
        for d in EMPTY_DIRS:
            z.writestr(f"reconciliation_poc/{d}/.keep", "")
    return target


if __name__ == "__main__":
    path = main()
    with zipfile.ZipFile(path) as z:
        print(f"{path} — {len(z.namelist())} fichiers")
