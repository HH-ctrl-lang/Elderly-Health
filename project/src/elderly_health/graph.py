"""Typed DOT graph prior with explicit roles and differentiable mask penalties."""

import re
from pathlib import Path


def parse_dot_graph(path):
    text = Path(path).read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*|#[^\n]*", "", text)
    node = r'"?([A-Za-z_][A-Za-z_0-9]*)"?'
    edges = set(re.findall(node + r"\s*->\s*" + node, text))
    if not edges:
        raise ValueError("Graph contains no supported directed edges")
    return edges, {n for edge in edges for n in edge}


def compute_roles(edges, target, features):
    parents = {src for src, dst in edges if dst == target}
    children = {dst for src, dst in edges if src == target}
    # Other ancestors may be relevant, but are not identified as confounders here.
    return {
        name: "parent" if name in parents else "child" if name in children else "unrelated"
        for name in features
        if name != target
    }


def load_graph_prior(path, target, features):
    edges, nodes = parse_dot_graph(path)
    if target not in nodes:
        raise ValueError(f"Graph target not found: {target}")
    unknown = nodes - set(features) - {target}
    if unknown:
        raise ValueError(f"Unmapped graph variables: {sorted(unknown)}")
    return compute_roles(edges, target, features)


def build_graph_mask_prior(roles, features):
    """Compatibility helper: all callers now pass role strings, never numeric codes."""
    return {str(name): roles.get(str(name), "unrelated") for name in features}


def graph_guided_loss(mask, feature_names, prior):
    import torch

    target_map = {"parent": (1.0, 1.0), "child": (0.0, 0.0), "unrelated": (0.0, 0.3)}
    terms = []
    for name in feature_names:
        role = prior.get(str(name), "unrelated")
        if role not in target_map:
            raise ValueError(f"Invalid graph role for {name}: {role!r}")
        terms.append(target_map[role])
    if len(terms) != mask.numel():
        raise ValueError("Feature names and gate dimensions differ")
    targets = torch.tensor([t for t, _ in terms], device=mask.device, dtype=mask.dtype)
    weights = torch.tensor([w for _, w in terms], device=mask.device, dtype=mask.dtype)
    active = weights > 0
    if not active.any():
        return mask.sum() * 0.0
    return ((mask[active] - targets[active]).square() * weights[active]).mean()


def prior_jaccard(mask, feature_names, prior):
    if prior is None:
        return None
    selected = {name for name, score in zip(feature_names, mask) if score > 0.5}
    parents = {name for name, role in prior.items() if role == "parent"}
    union = selected | parents
    return len(selected & parents) / len(union) if union else 0.0
