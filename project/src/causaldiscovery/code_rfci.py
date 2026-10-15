from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz

DATA_PATH = Path("data_mission2.csv")

ALPHA = 0.01

N_SUBSAMPLE = 50
SUBSAMPLE_RATIO = 0.80

EDGE_THRESHOLD = 0.40

RANDOM_STATE = 42

DROP_COLUMNS = []

# 分层（用于后处理，不是硬约束）
TIER_MAP = {
    "tier1": [
        "age",
        "gender",
        "residence",
    ],

    "tier2": [
        "education",
        "wealth_status",
        "marital_status",
        "living_alone",
        "chronic_disease_grp",
    ],

    "tier3": [
        "social_activity",
        "smartphone_use_duration",
        # "screen_time",
        "sleep_duration",
        "sleep_quality",
        "physical_activity",
        "smoking",
        "drinking",
    ],

    "tier4": [
        "depression_score",
        "cognitive_impairment",
        "IADL",
    ],
}


def log(message: str) -> None:

    timestamp = pd.Timestamp.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print(
        f"[{timestamp}] {message}",
        flush=True
    )


def load_and_prepare_data(
    data_path: Path = DATA_PATH,
) -> tuple[pd.DataFrame, np.ndarray]:

    df = pd.read_csv(data_path)

    df = df.drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )

    df = df.replace(
        "缺失",
        pd.NA
    )

    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    df = df.dropna().copy()

    # 标准化
    x = (df - df.mean()) / (df.std() + 1e-8)

    return df, x.values


def extract_pag_edges(
    graph,
    columns: Sequence[str],
):

    directed_edges = []
    bidirected_edges = []
    undirected_edges = []
    partial_edges = []

    nodes = graph.graph

    n = len(columns)

    for i in range(n):

        for j in range(i + 1, n):

            edge_ij = nodes[i, j]
            edge_ji = nodes[j, i]

            src = columns[i]
            dst = columns[j]

            # i --> j
            if edge_ij == -1 and edge_ji == 1:
                directed_edges.append((src, dst))

            # i <-- j
            elif edge_ij == 1 and edge_ji == -1:
                directed_edges.append((dst, src))

            # i <-> j
            elif edge_ij == 1 and edge_ji == 1:
                bidirected_edges.append((src, dst))

            # i --- j
            elif edge_ij == -1 and edge_ji == -1:
                undirected_edges.append((src, dst))

            # i o-> j
            elif edge_ij == 2 and edge_ji == 1:
                partial_edges.append((src, dst))

            # i <-o j
            elif edge_ij == 1 and edge_ji == 2:
                partial_edges.append((dst, src))

            # i o-o j
            elif edge_ij == 2 and edge_ji == 2:
                partial_edges.append((src, dst))

    return (
        directed_edges,
        bidirected_edges,
        undirected_edges,
        partial_edges,
    )


def run_rfci_once(
    x: np.ndarray,
    columns: Sequence[str],
    run_label: str = "RFCI",
):

    start = perf_counter()

    log(
        f"{run_label}: "
        f"start RFCI "
        f"shape={x.shape}"
    )

    g, edges = fci(
        dataset=x,
        independence_test_method=fisherz,
        alpha=ALPHA,
        depth=-1,
        verbose=False,
        show_progress=False,
    )

    elapsed = perf_counter() - start

    log(
        f"{run_label}: "
        f"finished "
        f"{elapsed:.1f}s"
    )

    return g


def run_subsample_rfci(
    x: np.ndarray,
    columns: Sequence[str],
):

    rng = np.random.default_rng(RANDOM_STATE)

    directed_counter = Counter()

    n = len(x)

    subsample_size = int(
        round(n * SUBSAMPLE_RATIO)
    )

    for run_idx in range(1, N_SUBSAMPLE + 1):

        log(
            f"Subsample "
            f"{run_idx:02d}/{N_SUBSAMPLE}"
        )

        idx = rng.choice(
            n,
            size=subsample_size,
            replace=False,
        )

        x_sub = x[idx]

        g = run_rfci_once(
            x_sub,
            columns,
            run_label=f"RFCI-{run_idx:02d}"
        )

        (
            directed_edges,
            bidirected_edges,
            undirected_edges,
            partial_edges,
        ) = extract_pag_edges(
            g,
            columns
        )

        directed_counter.update(
            directed_edges
        )

    return directed_counter


def build_stable_edges(
    directed_counter: Counter,
):

    stable_edges = []

    for edge, count in directed_counter.items():

        freq = count / N_SUBSAMPLE

        if freq >= EDGE_THRESHOLD:

            stable_edges.append(
                (
                    edge[0],
                    edge[1],
                    count,
                    round(freq, 3),
                )
            )

    stable_edges.sort(
        key=lambda x: x[3],
        reverse=True
    )

    return stable_edges


def apply_domain_filter(
    stable_edges,
):

    filtered = []

    forbidden = {

        # 明显错误方向
        ("drinking", "gender"),
        ("smoking", "gender"),
        ("education", "residence"),

        ("IADL", "age"),
        ("IADL", "gender"),

        ("depression_score", "gender"),
        ("depression_score", "age"),

        ("cognitive_impairment", "age"),
        ("cognitive_impairment", "gender"),

        # 高共线
        ("smartphone_use_duration", "screen_time"),
        ("screen_time", "smartphone_use_duration"),
    }

    for edge in stable_edges:

        src = edge[0]
        dst = edge[1]

        if (src, dst) in forbidden:
            continue

        filtered.append(edge)

    return filtered


def stable_edges_to_dot(
    stable_edges,
):

    parts = []

    for src, dst, count, freq in stable_edges:

        parts.append(
            f'"{src}"->"{dst}";'
        )

    return "digraph {" + " ".join(parts) + "}"


def save_stable_edges_csv(
    path: Path,
    stable_edges,
):

    df = pd.DataFrame(
        stable_edges,
        columns=[
            "source",
            "target",
            "count",
            "frequency",
        ]
    )

    df.to_csv(
        path,
        index=False
    )


def main():

    start_total = perf_counter()

    project_root = Path(__file__).resolve().parent

    data_path = project_root / DATA_PATH

    log("Loading data")

    df, x = load_and_prepare_data(data_path)

    columns = df.columns.tolist()

    log(
        f"Samples: {len(df)}"
    )

    log(
        f"Variables: {len(columns)}"
    )

    
    log("Running baseline RFCI")

    baseline_graph = run_rfci_once(
        x,
        columns,
        run_label="Baseline"
    )

    (
        directed_edges,
        bidirected_edges,
        undirected_edges,
        partial_edges,
    ) = extract_pag_edges(
        baseline_graph,
        columns
    )

    print("\n=== Baseline RFCI ===")

    print(
        f"Directed: {len(directed_edges)}"
    )

    print(
        f"Bidirected: {len(bidirected_edges)}"
    )

    print(
        f"Undirected: {len(undirected_edges)}"
    )

    print(
        f"Partial: {len(partial_edges)}"
    )

    
    log("Running subsampling RFCI")

    directed_counter = run_subsample_rfci(
        x,
        columns
    )

    stable_edges = build_stable_edges(
        directed_counter
    )

    stable_edges = apply_domain_filter(
        stable_edges
    )

    print("\n=== Stable Edges ===")

    for edge in stable_edges[:30]:

        print(edge)

    print(
        f"\nStable edge count: "
        f"{len(stable_edges)}"
    )

   
    stable_dot = stable_edges_to_dot(
        stable_edges
    )

    save_stable_edges_csv(
        project_root / "rfci_stable_edges.csv",
        stable_edges
    )

    (
        project_root / "rfci_stable_graph.dot"
    ).write_text(
        stable_dot,
        encoding="utf-8"
    )

    elapsed_total = (
        perf_counter() - start_total
    )

    log(
        f"Finished. "
        f"Elapsed={elapsed_total/60:.1f} min"
    )


if __name__ == "__main__":
    main()