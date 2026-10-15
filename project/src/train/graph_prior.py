"""
解析 causal_graph.dot → 提取每个特征相对于 target 的因果角色
用于 L_graph 结构正则化
"""
import os
import re
from typing import Dict, Set, Tuple, List


def parse_dot_graph(dot_path: str) -> Tuple[Set[Tuple[str, str]], Set[str]]:
    """
    解析 digraph 文件，返回 (directed_edges, all_nodes)
    directed_edges: {(src, dst), ...}
    """
    with open(dot_path, "r", encoding="utf-8") as f:
        text = f.read()

    edges: Set[Tuple[str, str]] = set()
    nodes: Set[str] = set()

    # 匹配 a -> b; 或 a -> b [label=...];
    pattern = r'(\w+)\s*->\s*(\w+)\s*(?:\[.*?\])?\s*;'
    for match in re.finditer(pattern, text):
        src, dst = match.group(1), match.group(2)
        edges.add((src, dst))
        nodes.add(src)
        nodes.add(dst)

    # 也匹配 a -> b (无分号，最后一行可能省略)
    pattern2 = r'(\w+)\s*->\s*(\w+)\s*$'
    for match in re.finditer(pattern2, text, re.MULTILINE):
        src, dst = match.group(1), match.group(2)
        edges.add((src, dst))
        nodes.add(src)
        nodes.add(dst)

    return edges, nodes


def compute_roles(
    edges: Set[Tuple[str, str]],
    target: str,
    features: List[str],
) -> Dict[str, str]:
    """
    根据因果图计算每个特征相对于 target 的角色。

    角色：
      "parent"     — 直接原因 (→ target)
      "child"      — target 的子节点 (target →)
      "confounder" — 与 target 通过无向边/共因关联 (不在图中显式表示时，
                      自动识别：既不是 parent 也不是 child 但通过路径相连)
      "unrelated"  — 与 target 无直接边

    返回: {feature_name: role}
    """
    parents: Set[str] = set()
    children: Set[str] = set()
    neighbors: Set[str] = set()

    for src, dst in edges:
        if dst == target:
            parents.add(src)
        if src == target:
            children.add(dst)
        if src == target or dst == target:
            neighbors.add(src if dst == target else dst)

    roles = {}
    for feat in features:
        if feat == target:
            continue
        if feat in parents:
            roles[feat] = "parent"
        elif feat in children:
            roles[feat] = "child"
        elif feat in neighbors:
            roles[feat] = "confounder"
        else:
            roles[feat] = "unrelated"

    return roles


def build_graph_mask_prior(
    roles: Dict[str, str],
    features: List[str],
) -> Dict[str, float]:
    """
    将角色映射转为 mask 先验权重。

    parent    → +1.0  (鼓励 mask → 1)
    child     →  0.0  (不加惩罚，让数据决定)
    confounder → -1.0  (鼓励 mask → 0)
    unrelated → -0.5  (轻微鼓励 mask → 0)
    """
    prior_map = {
        "parent": 1.0,
        "child": 0.0,
        "confounder": -1.0,
        "unrelated": -0.5,
    }

    prior = {}
    for feat in features:
        role = roles.get(feat, "unrelated")
        prior[feat] = prior_map[role]

    return prior


def graph_guided_loss(mask, feature_names, prior: Dict[str, float]):
    """
    图结构一致性损失。

    L_graph = mean( (mask_i - target_i)^2 )
    其中 target_i:
      parent:    1.0 (鼓励选中)
      confounder: 0.0 (鼓励排除)
      unrelated:  0.0 (鼓励排除)
      child:      no penalty (mask = current value → grad = 0)

    实现为：对 parent/confounder/unrelated 施加 MSE 惩罚，
    对 child 不施加惩罚。
    """
    import torch

    targets = []
    weights = []
    for i, name in enumerate(feature_names):
        name_str = str(name)
        role = prior.get(name_str, "unrelated")
        if role == "parent":
            targets.append(1.0)
            weights.append(1.0)
        elif role == "confounder":
            targets.append(0.0)
            weights.append(1.0)
        elif role == "unrelated":
            targets.append(0.0)
            weights.append(0.3)  # 弱惩罚
        elif role == "child":
            # child 不加惩罚
            targets.append(-1.0)  # dummy
            weights.append(0.0)

    targets_t = torch.tensor(targets, dtype=torch.float32, device=mask.device)
    weights_t = torch.tensor(weights, dtype=torch.float32, device=mask.device)

    # 只对 weight > 0 的特征计算损失
    active = weights_t > 0
    if active.sum() == 0:
        return torch.tensor(0.0, device=mask.device)

    loss = ((mask[active] - targets_t[active]) ** 2 * weights_t[active]).mean()
    return loss


def load_graph_prior(graph_path: str, target: str, features: List[str]) -> Dict[str, float]:
    """
    一站式：加载 .dot 文件 → 返回 mask 先验字典
    """
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"因果图文件不存在: {graph_path}")

    edges, nodes = parse_dot_graph(graph_path)
    roles = compute_roles(edges, target, features)
    prior = build_graph_mask_prior(roles, features)

    # 打印摘要
    print(f"\n=== 因果图先验 ===")
    print(f"节点: {len(nodes)}, 边: {len(edges)}")
    for role_name in ["parent", "child", "confounder", "unrelated"]:
        role_feats = [f for f, r in roles.items() if r == role_name]
        if role_feats:
            print(f"  {role_name}: {role_feats}")
    print(f"==================\n")

    return prior
