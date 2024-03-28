from pathlib import Path


def enable_debug_mode(path: Path) -> None:
    global _debug_dir
    _debug_dir = path
    path.mkdir(exist_ok=True, parents=True)


def get_debug_dir() -> Path:
    global _debug_dir
    assert _debug_dir is not None
    return _debug_dir


def debug_mode_enabled() -> bool:
    return _debug_dir is not None
