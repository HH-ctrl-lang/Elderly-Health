from __future__ import annotations

from collections import Counter
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from castle.algorithms import Notears
from castle.common.priori_knowledge import PrioriKnowledge


DATA_PATH = Path("data_analysis.xlsx")

N_SUBSAMPLE = 50
SUBSAMPLE_RATIO = 0.80
EDGE_THRESHOLD = 0.55
RANDOM_STATE = 42

DROP_COLUMNS = []

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
        "screen_time",
        "smartphone_use_duration",
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

# 可选强制边
REQUIRED_NAMED_EDGES = [
    # ("age", "cognitive_impairment"),
]


def log(message: str) -> None:
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_and_prepare_data(
    data_path: Path = DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    original_data = pd.read_excel(data_path, header=0)

    original_data = original_data.drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )

    original_data = original_data.replace(
        "缺失",
        pd.NA
    )

    original_data = original_data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    original_data_clean = original_data.dropna().copy()

    x_df = pd.DataFrame(
        original_data_clean,
        columns=original_data_clean.columns
    )

    return original_data_clean, x_df


def build_prior_knowledge(
    columns: Sequence[str],
):

    columns = list(columns)

    missing_vars = {
        tier_name: [
            var for var in vars_in_tier
            if var not in columns
        ]
        for tier_name, vars_in_tier in TIER_MAP.items()
    }

    missing_vars = {
        k: v for k, v in missing_vars.items() if v
    }

    if missing_vars:
        raise ValueError(
            f"Missing variables: {missing_vars}"
        )

    col_to_idx = {
        name: idx
        for idx, name in enumerate(columns)
    }

    forbidden_named_edges = set()

    for src, dst in product(columns, TIER_MAP["tier1"]):

        if src != dst:
            forbidden_named_edges.add((src, dst))


    # IADL 不可能影响年龄/性别
    impossible_upstream = [
        "age",
        "gender",
        "residence",
    ]

    for dst in impossible_upstream:

        forbidden_named_edges.add(("IADL", dst))
        forbidden_named_edges.add(("cognitive_impairment", dst))
        forbidden_named_edges.add(("depression_score", dst))

    # 婚姻状态不可能被 living_alone 决定
    forbidden_named_edges.add(
        ("living_alone", "marital_status")
    )

    # 吸烟饮酒方向不确定，禁止强制方向
    forbidden_named_edges.add(
        ("smoking", "drinking")
    )

    forbidden_named_edges.add(
        ("drinking", "smoking")
    )

    # 手机使用与屏幕时间本质重复
    forbidden_named_edges.add(
        ("screen_time", "smartphone_use_duration")
    )

    forbidden_named_edges.add(
        ("smartphone_use_duration", "screen_time")
    )

   
    missing_required_edges = [

        edge for edge in REQUIRED_NAMED_EDGES

        if edge[0] not in col_to_idx
        or edge[1] not in col_to_idx
    ]

    if missing_required_edges:
        raise ValueError(
            f"Missing required edges: {missing_required_edges}"
        )

    forbidden_edges = sorted(
        (col_to_idx[src], col_to_idx[dst])

        for src, dst in forbidden_named_edges
    )

    required_edges = sorted(
        (col_to_idx[src], col_to_idx[dst])

        for src, dst in REQUIRED_NAMED_EDGES
    )

    priori = PrioriKnowledge(len(columns))

    priori.add_forbidden_edges(forbidden_edges)

    priori.add_required_edges(required_edges)

    return (
        TIER_MAP,
        col_to_idx,
        forbidden_named_edges,
        forbidden_edges,
        priori,
    )


def run_notears_once(
    x_df: pd.DataFrame,
    priori: PrioriKnowledge,
    run_label: str = "NOTEARS",
) -> np.ndarray:

    start_time = perf_counter()

    log(
        f"{run_label}: start "
        f"shape={x_df.shape}"
    )

    # 标准化
    x_scaled = (
        (x_df - x_df.mean())
        / (x_df.std() + 1e-8)
    )

    # NOTEARS
    nt = Notears(
        loss_type="l2",
        w_threshold=0.15,
    )

    nt.learn(
        x_scaled,
        p_knowledge=priori.matrix
    )

    causal_matrix = nt.causal_matrix.copy()

    forbidden_mask = (priori.matrix == -1)

    causal_matrix[forbidden_mask] = 0

    elapsed = perf_counter() - start_time

    log(
        f"{run_label}: finished "
        f"in {elapsed:.1f}s"
    )

    return causal_matrix


def extract_edges_from_matrix(
    matrix: np.ndarray,
    columns: Sequence[str],
):

    directed_edges = []
    undirected_edges = []

    seen_undirected = set()

    rows, cols = matrix.shape

    for i in range(rows):

        for j in range(cols):

            if matrix[i, j] != 1:
                continue

            # 双向边
            if matrix[j, i] == 1:

                edge = tuple(sorted((i, j)))

                if i != j and edge not in seen_undirected:

                    seen_undirected.add(edge)

                    undirected_edges.append(
                        tuple(sorted((columns[i], columns[j])))
                    )

            else:

                directed_edges.append(
                    (columns[i], columns[j])
                )

    directed_edges.sort()
    undirected_edges.sort()

    return directed_edges, undirected_edges


def run_subsample_notears(
    x_df: pd.DataFrame,
    n_subsample: int,
    subsample_ratio: float,
    random_state: int,
    priori: PrioriKnowledge,
):

    rng = np.random.default_rng(random_state)

    directed_counts = Counter()
    undirected_counts = Counter()

    n_samples = len(x_df)

    subsample_size = max(
        2,
        int(round(n_samples * subsample_ratio))
    )

    for run_idx in range(1, n_subsample + 1):

        log(
            f"Subsample "
            f"{run_idx:02d}/{n_subsample}"
        )

        sample_idx = rng.choice(
            n_samples,
            size=subsample_size,
            replace=False
        )

        x_sub = x_df.iloc[sample_idx].reset_index(drop=True)

        matrix = run_notears_once(
            x_sub,
            priori,
            run_label=f"Subsample {run_idx:02d}"
        )

        directed_edges, undirected_edges = (
            extract_edges_from_matrix(
                matrix,
                x_df.columns.tolist()
            )
        )

        directed_counts.update(directed_edges)

        undirected_counts.update(undirected_edges)

    return directed_counts, undirected_counts


def build_frequency_tables(
    directed_counts: Counter,
    undirected_counts: Counter,
    n_runs: int,
):

    records = []

    for (src, dst), count in directed_counts.items():

        records.append({
            "edge_type": "directed",
            "source": src,
            "target": dst,
            "count": count,
            "frequency": count / n_runs,
        })

    for (src, dst), count in undirected_counts.items():

        records.append({
            "edge_type": "undirected",
            "source": src,
            "target": dst,
            "count": count,
            "frequency": count / n_runs,
        })

    edge_frequency_df = pd.DataFrame(records)

    if edge_frequency_df.empty:

        return pd.DataFrame(
            columns=[
                "edge_type",
                "source",
                "target",
                "count",
                "frequency",
            ]
        )

    return edge_frequency_df.sort_values(
        by=[
            "frequency",
            "count",
        ],
        ascending=False,
    ).reset_index(drop=True)


def build_stable_graph(
    edge_frequency_df: pd.DataFrame,
    columns: Sequence[str],
    edge_threshold: float,
):

    pair_candidates = {}

    stable_df = edge_frequency_df.loc[
        edge_frequency_df["frequency"] >= edge_threshold
    ].copy()

    for row in stable_df.to_dict("records"):

        pair_key = tuple(
            sorted((row["source"], row["target"]))
        )

        pair_candidates.setdefault(
            pair_key,
            []
        ).append(row)

    chosen_directed_edges = []
    chosen_undirected_edges = []

    for pair, candidates in pair_candidates.items():

        best = max(
            candidates,
            key=lambda row: float(row["frequency"])
        )

        chosen_directed_edges.append(
            (best["source"], best["target"])
        )

    col_to_idx = {
        name: idx
        for idx, name in enumerate(columns)
    }

    stable_matrix = np.zeros(
        (len(columns), len(columns)),
        dtype=int
    )

    for src, dst in chosen_directed_edges:

        stable_matrix[
            col_to_idx[src],
            col_to_idx[dst]
        ] = 1

    return (
        chosen_directed_edges,
        chosen_undirected_edges,
        stable_matrix,
    )


def matrix_to_dot(
    matrix: np.ndarray,
    columns: Sequence[str],
) -> str:

    directed_edges, undirected_edges = (
        extract_edges_from_matrix(
            matrix,
            columns
        )
    )

    parts = [
        f'"{src}"->"{dst}";'
        for src, dst in directed_edges
    ]

    parts.extend(
        f'"{src}"->"{dst}"; "{dst}"->"{src}";'
        for src, dst in undirected_edges
    )

    return "digraph {" + " ".join(parts) + "}"


def save_edges_csv(
    path: Path,
    edges,
    edge_type: str,
):

    df = pd.DataFrame(
        edges,
        columns=["source", "target"]
    )

    df.insert(
        0,
        "edge_type",
        edge_type
    )

    df.to_csv(path, index=False)


def main():

    script_start = perf_counter()

    project_root = Path(__file__).resolve().parent

    data_path = project_root / DATA_PATH

    log("Starting NOTEARS pipeline")

    original_data_clean, x_df = (
        load_and_prepare_data(data_path)
    )

    (
        tier_map,
        col_to_idx,
        forbidden_named_edges,
        forbidden_edges,
        priori,
    ) = build_prior_knowledge(x_df.columns)

    log(
        f"Samples: {len(original_data_clean)}"
    )

    log(
        f"Variables: {len(x_df.columns)}"
    )


    log("Running baseline NOTEARS")

    baseline_matrix = run_notears_once(
        x_df,
        priori,
        run_label="Baseline"
    )

    baseline_directed_edges, baseline_undirected_edges = (
        extract_edges_from_matrix(
            baseline_matrix,
            x_df.columns.tolist()
        )
    )

    print("\n=== Baseline ===")

    print(
        f"Directed edges: "
        f"{len(baseline_directed_edges)}"
    )

    log("Running subsampling")

    directed_counts, undirected_counts = (
        run_subsample_notears(
            x_df=x_df,
            n_subsample=N_SUBSAMPLE,
            subsample_ratio=SUBSAMPLE_RATIO,
            random_state=RANDOM_STATE,
            priori=priori,
        )
    )

    edge_frequency_df = build_frequency_tables(
        directed_counts,
        undirected_counts,
        N_SUBSAMPLE
    )

    print("\n=== Top Frequency Edges ===")

    print(
        edge_frequency_df.head(20).to_string(index=False)
    )

    stable_directed_edges, stable_undirected_edges, stable_matrix = (
        build_stable_graph(
            edge_frequency_df,
            x_df.columns.tolist(),
            EDGE_THRESHOLD
        )
    )

    print("\n=== Stable Graph ===")

    print(
        f"Stable directed edges: "
        f"{len(stable_directed_edges)}"
    )

    stable_dot_text = matrix_to_dot(
        stable_matrix,
        x_df.columns.tolist()
    )

    save_edges_csv(
        project_root / "notears_baseline_directed_edges.csv",
        baseline_directed_edges,
        "directed"
    )

    edge_frequency_df.to_csv(
        project_root / "notears_subsample_edge_frequency.csv",
        index=False
    )

    save_edges_csv(
        project_root / "notears_subsample_stable_directed_edges.csv",
        stable_directed_edges,
        "directed"
    )

    (project_root / "notears_subsample_stable_graph.dot").write_text(
        stable_dot_text,
        encoding="utf-8"
    )

    total_elapsed = perf_counter() - script_start

    log(
        f"Finished. "
        f"Elapsed={total_elapsed/60:.1f}min"
    )


if __name__ == "__main__":
    main()