"""Compatibility exports; maintained graph handling lives in elderly_health.graph."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elderly_health.graph import (  # noqa: E402,F401
    build_graph_mask_prior,
    compute_roles,
    graph_guided_loss,
    load_graph_prior,
    parse_dot_graph,
)
