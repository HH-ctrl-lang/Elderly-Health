"""Run from any working directory: python /path/to/project/run.py --help."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from elderly_health.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
