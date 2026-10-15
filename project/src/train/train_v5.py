"""Compatibility entry point. Use project/run.py for the maintained implementation.

Historical source is preserved in project/legacy/. --output is now required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elderly_health.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["train", "--model", "v5", *sys.argv[1:]])
