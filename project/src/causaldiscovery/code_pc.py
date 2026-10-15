from __future__ import annotations

from collections import Counter
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from castle.algorithms import PC
from castle.common.priori_knowledge import PrioriKnowledge


DATA_PATH = Path("data_analysis.xlsx")

# ===== 可调参数 =====
N_SUBSAMPLE = 20           # 重复多少轮
SUBSAMPLE_RATIO = 0.80     # 每轮抽取原样本的 80%
EDGE_THRESHOLD = 0.80      # 稳定边阈值
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

REQUIRED_NAMED_EDGES = [
    # ("social_activity", "IALD"),
]


def log(message: str) -> None:
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_and_prepare_data(data_path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    original_data = pd.read_excel(data_path, header=0)
    original_data = original_data.drop(columns=DROP_COLUMNS, errors="ignore")
    original_data = original_data.replace("缺失", pd.NA)
    original_data = original_data.apply(
    pd.to_numeric,
    errors="coerce"
    )

    original_data_clean = (
        original_data
        .dropna()
        .copy()
    )

    x_df = pd.DataFrame(original_data_clean, columns=original_data_clean.columns)
    return original_data_clean, x_df


def build_prior_knowledge(
    columns: Sequence[str],
) -> tuple[dict[str, list[str]], dict[str, int], set[tuple[str, str]], list[tuple[int, int]], PrioriKnowledge]:
    columns = list(columns)

    missing_vars = {
        tier_name: [var for var in vars_in_tier if var not in columns]
        for tier_name, vars_in_tier in TIER_MAP.items()
    }
    missing_vars = {k: v for k, v in missing_vars.items() if v}
    if missing_vars:
        raise ValueError(f"These tier variables are missing from columns: {missing_vars}")

    col_to_idx = {name: idx for idx, name in enumerate(columns)}
    forbidden_named_edges: set[tuple[str, str]] = set()
    missing_required_edges = [
        edge for edge in REQUIRED_NAMED_EDGES
        if edge[0] not in col_to_idx or edge[1] not in col_to_idx
    ]
    if missing_required_edges:
        raise ValueError(f"These required edges reference missing columns: {missing_required_edges}")

    # 任意变量都不能指向 tier1
    for src, dst in product(columns, TIER_MAP["tier1"]):
        if src != dst:
            forbidden_named_edges.add((src, dst))


    # 健康/行为变量不能指向社会人口学早期变量
    forbidden_parents = [
        "chronic_disease_grp",
        "social_activity",
        "smartphone_use_duration",
        "screen_time",
        "IADL",
        "depression_score",
        "cognitive_impairment",
        "sleep_quality",
        "sleep_duration",
        "physical_activity",
        "smoking",
        "drinking",
    ]

    forbidden_targets = [
        "education",
        "marital_status",
        "living_alone",
    ]

    for src, dst in product(forbidden_parents, forbidden_targets):
        if src != dst:
            forbidden_named_edges.add((src, dst))


    # smartphone 与 screen_time 本质重复，禁止互相指向
    forbidden_named_edges.add(
        ("screen_time", "smartphone_use_duration")
    )

    forbidden_named_edges.add(
        ("smartphone_use_duration", "screen_time")
    )

    #禁止 IADL 指向明显上游变量
    upstream_vars = [
        "age",
        "gender",
        "education",
        "marital_status",
        "living_alone",
        "wealth_status",
    ]

    for dst in upstream_vars:
        forbidden_named_edges.add(("IADL", dst))

    # 禁止吸烟与饮酒互为因果
    forbidden_named_edges.add(("smoking", "drinking"))
    forbidden_named_edges.add(
        ("living_alone", "marital_status")
    )


    forbidden_edges = sorted((col_to_idx[src], col_to_idx[dst]) for src, dst in forbidden_named_edges)
    required_edges = sorted((col_to_idx[src], col_to_idx[dst]) for src, dst in REQUIRED_NAMED_EDGES)

    priori = PrioriKnowledge(len(columns))
    priori.add_forbidden_edges(forbidden_edges)
    priori.add_required_edges(required_edges)

    return TIER_MAP, col_to_idx, forbidden_named_edges, forbidden_edges, priori


def run_pc_once(
    x_df: pd.DataFrame,
    priori: PrioriKnowledge,
    run_label: str = "PC",
) -> np.ndarray:
    start_time = perf_counter()
    log(f"{run_label}: start pc.learn() on shape={x_df.shape}")
    pc = PC(variant="parallel", ci_test="chi2", priori_knowledge=priori)
    pc.learn(x_df, p_cores=4)
    elapsed = perf_counter() - start_time
    log(f"{run_label}: finished pc.learn() in {elapsed:.1f}s")
    return pc.causal_matrix.copy()


def extract_edges_from_matrix(
    matrix: np.ndarray,
    columns: Sequence[str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    directed_edges: list[tuple[str, str]] = []
    undirected_edges: list[tuple[str, str]] = []
    seen_undirected: set[tuple[int, int]] = set()

    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] != 1:
                continue
            if matrix[j, i] == 1:
                edge = tuple(sorted((i, j)))
                if i != j and edge not in seen_undirected:
                    seen_undirected.add(edge)
                    undirected_edges.append(tuple(sorted((columns[i], columns[j]))))
            else:
                directed_edges.append((columns[i], columns[j]))

    directed_edges.sort()
    undirected_edges.sort()
    return directed_edges, undirected_edges


def run_subsample_pc(
    x_df: pd.DataFrame,
    n_subsample: int,
    subsample_ratio: float,
    random_state: int,
    priori: PrioriKnowledge,
) -> tuple[Counter, Counter]:
    if not (0 < subsample_ratio <= 1):
        raise ValueError("subsample_ratio must be in (0, 1].")

    rng = np.random.default_rng(random_state)
    directed_counts: Counter = Counter()
    undirected_counts: Counter = Counter()

    n_samples = len(x_df)
    subsample_size = max(2, int(round(n_samples * subsample_ratio)))
    subsample_start = perf_counter()

    for run_idx in range(1, n_subsample + 1):
        iter_start = perf_counter()

        log(
            f"Subsample {run_idx:02d}/{n_subsample}: "
            f"sampling rows WITHOUT replacement, size={subsample_size}/{n_samples}"
        )

        # 不放回抽样：不会人为制造新的重复行
        sample_idx = rng.choice(n_samples, size=subsample_size, replace=False)
        x_sub = x_df.iloc[sample_idx].reset_index(drop=True)

        # 这里统计一下“天然重复”还剩多少，方便你观察
        n_unique_rows = x_sub.drop_duplicates().shape[0]
        log(
            f"Subsample {run_idx:02d}/{n_subsample}: "
            f"rows={len(x_sub)}, unique_rows={n_unique_rows}, duplicate_rows={len(x_sub) - n_unique_rows}"
        )

        matrix = run_pc_once(x_sub, priori, run_label=f"Subsample {run_idx:02d}/{n_subsample}")
        directed_edges, undirected_edges = extract_edges_from_matrix(matrix, x_df.columns.tolist())

        directed_counts.update(directed_edges)
        undirected_counts.update(undirected_edges)

        iter_elapsed = perf_counter() - iter_start
        elapsed_total = perf_counter() - subsample_start
        avg_elapsed = elapsed_total / run_idx
        remaining = avg_elapsed * (n_subsample - run_idx)

        log(
            f"Subsample {run_idx:02d}/{n_subsample}: "
            f"directed={len(directed_edges)}, undirected={len(undirected_edges)}, "
            f"iter_time={iter_elapsed:.1f}s, elapsed={elapsed_total/60:.1f}min, eta={remaining/60:.1f}min"
        )

    return directed_counts, undirected_counts


def build_frequency_tables(
    directed_counts: Counter,
    undirected_counts: Counter,
    n_runs: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for (src, dst), count in directed_counts.items():
        records.append(
            {
                "edge_type": "directed",
                "source": src,
                "target": dst,
                "count": count,
                "frequency": count / n_runs,
            }
        )

    for (src, dst), count in undirected_counts.items():
        records.append(
            {
                "edge_type": "undirected",
                "source": src,
                "target": dst,
                "count": count,
                "frequency": count / n_runs,
            }
        )

    edge_frequency_df = pd.DataFrame(records)
    if edge_frequency_df.empty:
        return pd.DataFrame(columns=["edge_type", "source", "target", "count", "frequency"])

    return edge_frequency_df.sort_values(
        by=["frequency", "count", "edge_type", "source", "target"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)


def _pair_key(src: str, dst: str) -> tuple[str, str]:
    return tuple(sorted((src, dst)))


def build_stable_graph(
    edge_frequency_df: pd.DataFrame,
    columns: Sequence[str],
    edge_threshold: float,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], np.ndarray]:
    pair_candidates: dict[tuple[str, str], list[dict[str, object]]] = {}

    stable_df = edge_frequency_df.loc[edge_frequency_df["frequency"] >= edge_threshold].copy()
    for row in stable_df.to_dict("records"):
        pair_candidates.setdefault(_pair_key(row["source"], row["target"]), []).append(row)

    chosen_directed_edges: list[tuple[str, str]] = []
    chosen_undirected_edges: list[tuple[str, str]] = []

    for pair, candidates in pair_candidates.items():
        best = max(
            candidates,
            key=lambda row: (
                float(row["frequency"]),
                1 if row["edge_type"] == "undirected" else 0,
            ),
        )

        top_tied = [row for row in candidates if float(row["frequency"]) == float(best["frequency"])]
        top_tied_types = {row["edge_type"] for row in top_tied}

        if "undirected" in top_tied_types and any(row["edge_type"] == "directed" for row in top_tied):
            chosen_undirected_edges.append(pair)
            continue

        if len(top_tied) > 1 and all(row["edge_type"] == "directed" for row in top_tied):
            chosen_undirected_edges.append(pair)
            continue

        if best["edge_type"] == "undirected":
            chosen_undirected_edges.append(pair)
        else:
            chosen_directed_edges.append((best["source"], best["target"]))

    chosen_directed_edges.sort()
    chosen_undirected_edges.sort()

    col_to_idx = {name: idx for idx, name in enumerate(columns)}
    stable_matrix = np.zeros((len(columns), len(columns)), dtype=int)

    for src, dst in chosen_directed_edges:
        stable_matrix[col_to_idx[src], col_to_idx[dst]] = 1

    for src, dst in chosen_undirected_edges:
        stable_matrix[col_to_idx[src], col_to_idx[dst]] = 1
        stable_matrix[col_to_idx[dst], col_to_idx[src]] = 1

    return chosen_directed_edges, chosen_undirected_edges, stable_matrix


def check_forbidden_violations(
    matrix: np.ndarray,
    forbidden_edges: Iterable[tuple[int, int]],
    columns: Sequence[str],
) -> list[tuple[str, str]]:
    violations: list[tuple[str, str]] = []
    for src_idx, dst_idx in forbidden_edges:
        if matrix[src_idx, dst_idx] == 1:
            violations.append((columns[src_idx], columns[dst_idx]))
    return sorted(violations)


def matrix_to_dot(matrix: np.ndarray, columns: Sequence[str]) -> str:
    directed_edges, undirected_edges = extract_edges_from_matrix(matrix, columns)
    parts = [f"{src}->{dst};" for src, dst in directed_edges]
    parts.extend(f"{src}->{dst}; {dst}->{src};" for src, dst in undirected_edges)
    return "digraph {" + " ".join(parts) + "}"


def save_edges_csv(path: Path, edges: Sequence[tuple[str, str]], edge_type: str) -> None:
    if edge_type not in {"directed", "undirected"}:
        raise ValueError("edge_type must be 'directed' or 'undirected'")

    df = pd.DataFrame(edges, columns=["source", "target"])
    df.insert(0, "edge_type", edge_type)
    df.to_csv(path, index=False)


def save_matrix_csv(path: Path, matrix: np.ndarray, columns: Sequence[str]) -> None:
    pd.DataFrame(matrix, index=columns, columns=columns).to_csv(path, index=True)


def main() -> None:
    script_start = perf_counter()
    project_root = Path(__file__).resolve().parent
    data_path = project_root / DATA_PATH

    log("Starting subsampling PC pipeline")
    log(f"Reading data from: {data_path}")

    original_data_clean, x_df = load_and_prepare_data(data_path)
    tier_map, col_to_idx, forbidden_named_edges, forbidden_edges, priori = build_prior_knowledge(x_df.columns)

    log(f"Samples after cleaning: {len(original_data_clean)}")
    log(f"Variables ({len(x_df.columns)}): {x_df.columns.tolist()}")

    print("\n=== Data Types ===")

    print(x_df.dtypes)

    print("\n=== Unique Values ===")

    for col in x_df.columns:

        print(f"\n{col}")

        print(sorted(x_df[col].dropna().unique()))

    log(f"Tier map: {tier_map}")
    log(f"Forbidden edges: {len(forbidden_edges)}")
    log(f"Column to index mapping: {col_to_idx}")

    log("Running baseline PC")
    baseline_matrix = run_pc_once(x_df, priori, run_label="Baseline")
    baseline_directed_edges, baseline_undirected_edges = extract_edges_from_matrix(
        baseline_matrix, x_df.columns.tolist()
    )
    baseline_violations = check_forbidden_violations(
        baseline_matrix, forbidden_edges, x_df.columns.tolist()
    )

    print("\n=== Baseline PC ===", flush=True)
    print(f"Directed edges: {len(baseline_directed_edges)}", flush=True)
    print(f"Undirected edges: {len(baseline_undirected_edges)}", flush=True)
    print(f"Forbidden violations: {baseline_violations}", flush=True)

    log(
        f"Starting subsampling loop with n_subsample={N_SUBSAMPLE}, "
        f"subsample_ratio={SUBSAMPLE_RATIO}, threshold={EDGE_THRESHOLD}"
    )
    directed_counts, undirected_counts = run_subsample_pc(
        x_df=x_df,
        n_subsample=N_SUBSAMPLE,
        subsample_ratio=SUBSAMPLE_RATIO,
        random_state=RANDOM_STATE,
        priori=priori,
    )

    edge_frequency_df = build_frequency_tables(
        directed_counts=directed_counts,
        undirected_counts=undirected_counts,
        n_runs=N_SUBSAMPLE,
    )
    stable_directed_edges, stable_undirected_edges, stable_matrix = build_stable_graph(
        edge_frequency_df=edge_frequency_df,
        columns=x_df.columns.tolist(),
        edge_threshold=EDGE_THRESHOLD,
    )
    stable_violations = check_forbidden_violations(
        stable_matrix, forbidden_edges, x_df.columns.tolist()
    )

    print("\n=== Subsampling Frequency Table (Top 20) ===", flush=True)
    if edge_frequency_df.empty:
        print("No edges were found across subsampling runs.", flush=True)
    else:
        print(edge_frequency_df.head(20).to_string(index=False), flush=True)

    print("\n=== Stable Graph ===", flush=True)
    print(f"Stable directed edges: {len(stable_directed_edges)}", flush=True)
    print(f"Stable undirected edges: {len(stable_undirected_edges)}", flush=True)
    print(f"Stable graph forbidden violations: {stable_violations}", flush=True)

    baseline_dot_text = matrix_to_dot(baseline_matrix, x_df.columns.tolist())
    stable_dot_text = matrix_to_dot(stable_matrix, x_df.columns.tolist())

    print("\n=== Stable Graph DOT ===", flush=True)
    print(stable_dot_text, flush=True)

    save_edges_csv(project_root / "pc_baseline_directed_edges.csv", baseline_directed_edges, "directed")
    save_edges_csv(project_root / "pc_baseline_undirected_edges.csv", baseline_undirected_edges, "undirected")
    edge_frequency_df.to_csv(project_root / "pc_subsample_edge_frequency.csv", index=False)
    save_edges_csv(
        project_root / "pc_subsample_stable_directed_edges.csv",
        stable_directed_edges,
        "directed",
    )
    save_edges_csv(
        project_root / "pc_subsample_stable_undirected_edges.csv",
        stable_undirected_edges,
        "undirected",
    )
    save_matrix_csv(project_root / "pc_subsample_stable_matrix.csv", stable_matrix, x_df.columns.tolist())

    (project_root / "pc_baseline_graph.dot").write_text(
        baseline_dot_text,
        encoding="utf-8",
    )
    (project_root / "pc_subsample_stable_graph.dot").write_text(
        stable_dot_text,
        encoding="utf-8",
    )

    total_elapsed = perf_counter() - script_start
    log(f"Output files written to project root. Total elapsed={total_elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
