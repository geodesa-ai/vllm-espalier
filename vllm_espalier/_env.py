"""Environment switches (ESPALIER_*)."""
import os


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)
