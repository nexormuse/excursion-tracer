"""풀이 쪽 코드(stats/, agent/, llm/)가 정답 파일을 읽거나 정답 관련 모듈을 import하지 못하게 막는다.

- 세 패키지와, 거기서 (직접·간접으로) import하는 프로젝트 내부 모듈 전체를 따라간다.
- 그 안에 excursion_tracer.eval import가 있거나, 'ground_truth' 문자열이 있으면 실패한다.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import pkgutil
from pathlib import Path

import pytest

import excursion_tracer

PKG_ROOT = Path(excursion_tracer.__file__).resolve().parent
PKG = "excursion_tracer"
SOLVER_PACKAGES = ("stats", "agent", "llm")
FORBIDDEN_MODULE_PREFIXES = (f"{PKG}.eval", f"{PKG}.sim.generate")
FORBIDDEN_TEXT = ("ground_truth", "groundtruth", "ground-truth")


def _module_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _module_file(name: str, root: Path) -> Path | None:
    rel = Path(*name.split(".")[1:])
    for cand in (root / rel.with_suffix(".py"), root / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _imports(path: Path, root: Path) -> set[str]:
    """파일 안의 모든 import를 절대 모듈 이름으로 돌려준다 (함수 안 import 포함)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mod = _module_name(path, root)
    is_pkg = path.name == "__init__.py"
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = mod.split(".")
                base = base if is_pkg else base[:-1]
                base = base[: len(base) - (node.level - 1)]
                prefix = ".".join(base + ([node.module] if node.module else []))
            else:
                prefix = node.module or ""
            out.add(prefix)
            out.update(f"{prefix}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if name in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    out.add(arg.value)
    return out


def violations(root: Path, packages=SOLVER_PACKAGES) -> list[str]:
    """풀이 쪽 패키지에서 도달 가능한 모든 내부 모듈을 검사해 위반 목록을 돌려준다."""
    pkg = root.name
    start = [p for sub in packages for p in (root / sub).rglob("*.py")]
    seen: set[Path] = set()
    queue = list(start)
    problems: list[str] = []
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        text = path.read_text(encoding="utf-8").lower()
        for word in FORBIDDEN_TEXT:
            if word in text:
                problems.append(f"{path.relative_to(root)}: '{word}' 문자열")
        for name in _imports(path, root):
            if not (name == pkg or name.startswith(pkg + ".")):
                continue
            name = name.replace(pkg, PKG, 1)
            if name.startswith(FORBIDDEN_MODULE_PREFIXES):
                problems.append(f"{path.relative_to(root)}: {name} import")
            target = _module_file(name, root)
            if target is not None and target not in seen:
                queue.append(target)
    return problems


def test_solver_packages_do_not_touch_ground_truth():
    assert violations(PKG_ROOT) == []


def test_solver_packages_exist():
    for sub in SOLVER_PACKAGES:
        assert (PKG_ROOT / sub / "__init__.py").is_file()


# ---------------------------------------------------------------- 검사기 자체 시험

def _fake_package(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / PKG
    for sub in ("", *SOLVER_PACKAGES, "eval", "sim", "util"):
        (root / sub).mkdir(parents=True, exist_ok=True)
        (root / sub / "__init__.py").write_text("")
    for rel, text in files.items():
        (root / rel).write_text(text)
    return root


@pytest.mark.parametrize(
    "files",
    [
        {"stats/a.py": "from excursion_tracer.eval import match\n"},
        {"agent/a.py": "from ..eval.ground import load\n"},
        {"llm/a.py": "import excursion_tracer.eval.metrics as m\n"},
        {"stats/a.py": "def f():\n    import excursion_tracer.eval\n"},
        {"stats/a.py": "import importlib\nm = importlib.import_module('excursion_tracer.eval')\n"},
        {"agent/a.py": "p = d / 'ground_truth.json'\n"},
        {"stats/a.py": "from excursion_tracer.sim.generate import GROUND_TRUTH_FILE\n"},
        # 간접 경로: stats → util → eval
        {"stats/a.py": "from excursion_tracer.util import helper\n",
         "util/helper.py": "from excursion_tracer.eval import x\n"},
        # 간접 경로: agent → util 모듈이 정답 파일 이름을 가짐
        {"agent/a.py": "from ..util.paths import P\n",
         "util/paths.py": "P = 'Ground_Truth.json'\n"},
    ],
)
def test_checker_catches_violations(tmp_path, files):
    assert violations(_fake_package(tmp_path, files))


def test_checker_allows_clean_code(tmp_path):
    root = _fake_package(
        tmp_path,
        {
            "stats/a.py": "import numpy as np\nfrom excursion_tracer.config import load_config\n",
            "agent/a.py": "from ..stats import a\nfrom excursion_tracer.sim.fab import Fab\n",
        },
    )
    assert violations(root) == []


def test_importing_solver_packages_does_not_open_ground_truth(monkeypatch):
    """풀이 쪽 모듈을 import하는 동안 정답 파일을 여는 코드가 실행되지 않는다."""
    real_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if "ground_truth" in str(file).lower():
            raise AssertionError(f"정답 파일 열기 시도: {file}")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    for sub in SOLVER_PACKAGES:
        pkg = importlib.import_module(f"{PKG}.{sub}")
        for info in pkgutil.walk_packages(pkg.__path__, prefix=f"{PKG}.{sub}."):
            importlib.import_module(info.name)
