import torch
import numpy as np
from collections import defaultdict
from typing import Optional

# =========================================================
# Dummy AlgoConfig (unchanged behavior)
# =========================================================
class AlgoConfig(dict):
    def get(self, k, default=None):
        return super().get(k, default)


# =========================================================
# Version 1: propagation + scores_add
# =========================================================
def compute_grpo_outcome_advantage_v1(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    current_index: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    step:  Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1)

    if depth is None:
        depth = np.ones_like(index)

    id2score = defaultdict(list)
    id2multiroll = defaultdict(list)
    propagation = defaultdict(list)
    id2mean, id2std = {}, {}

    u = min(10, len(depth))
    assert len(scores) >= u
    print(
        f"Computing GRPO advantage...\n\n"
        f"index: {index[:u]}\n"
        f"current_index: {current_index[:u] if current_index is not None else None}\n"
        f"depth: {depth[:u]}\n"
        f"scores: {scores[:u]}\n"
        f"scores[0]: {scores[0]}\n"
    )

    assert len(index) == len(depth), (
        "index, current_index and depth must have the same length."
    )

    if config.get("grpo_child_score_merge_fn", "max") == "max":
        merge_fn = torch.max
    elif config.get("grpo_child_score_merge_fn", "max") == "mean":
        merge_fn = torch.mean
    elif config.get("grpo_child_score_merge_fn", "max") == "min":
        merge_fn = torch.min

    with torch.no_grad():
        calculate_order = list(range(len(index)))
        calculate_order.sort(key=lambda x: -depth[x])

        current_order = depth[calculate_order[0]]
        current_pos = 0
        max_depth = current_order

        scores_add = torch.zeros_like(scores)
        c1 = sum([torch.eq(scores[i], -1) for i in range(len(index))])
        c0 = sum([torch.eq(scores[i], 0) for i in range(len(index))])

        if config.get("leaf_score_only", False):
            print("Using leaf score only mode in GRPO advantage computation.")
            for i in range(len(index)):
                if depth[i] < max_depth:
                    scores[i] = 0.0

        scale_x = config.get("qe_weight", 1.0)
        scale_y = config.get("merge_weight", 1.0) 
        if step >= 51 and config.get("remove_runtime_qe", False):
            print("Step > 51, so make scale = 0.0")
            scale_x = 0

        add_mask = np.zeros(len(index), dtype=bool)
        while current_order >= 1:
            current_depth_indices = []
            while (
                current_pos < len(calculate_order)
                and depth[calculate_order[current_pos]] == current_order
            ):
                current_depth_indices.append(calculate_order[current_pos])
                current_pos += 1

            current_order -= 1
            bsz = len(current_depth_indices)
            if depth[current_depth_indices[0]] < max_depth and current_index is not None:
                for i in range(bsz):
                    idx = current_depth_indices[i]
                    child_idx = current_index[idx]
                    if child_idx in propagation:
                        vals = [
                            t for t in propagation[child_idx]
                            if t.item() > 0
                        ]

                        scores_add[idx] = merge_fn(torch.stack(vals)) if vals else scores[idx].clone()
                        id2multiroll[index[idx]].append(
                            {"task_reward": scores[idx].item(), "sum": (scale_x * scores[idx] + scale_y * scores_add[idx]).item(), "propagation": vals}
                        )
                    elif depth[idx] != max_depth:
                        id2multiroll[index[idx]].append(
                            {"task_reward": scores[idx].item(), "sum": scores[idx].item(), "propagation": []}
                        )
                        add_mask[idx] = True
                        

            for i in range(bsz):
                idx = current_depth_indices[i]

                propagation[index[idx]].append(
                    scores_add[idx].clone() if depth[idx] != max_depth else scores[idx].clone()
                )

                if depth[idx] != max_depth and add_mask[idx] == False:
                    scores[idx] = scale_x * scores[idx] + scale_y * scores_add[idx]


        
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])


        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"No score in prompt index: {idx}")
        print(scores[:u])
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (
                    (scores[i] - id2mean[index[i]])
                    / (id2std[index[i]] + epsilon)
                )
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        print(f"adv: {scores[:u]}")
        scores = scores.unsqueeze(-1) * response_mask

        for key, li in id2multiroll.items():
            print(f"\nGroup {key} ({len(li)} rolls)")
            print(f"Mean: {id2mean[key]}  Std: {id2std[key]}")
            for i, entry in enumerate(li):
                print(f"  Roll {i}:")
                print(f"    task_reward = {entry['task_reward']}")
                print(f"    sum = {entry['sum']}")
                print(f"    propagation = {entry['propagation']}")
        print(
            f"check -1 "
            f"{c1}/{len(index)}\n"
            f"check 0 "
            f"{c0}/{len(index)}\n"
        )

    return scores, scores



# =========================================================
# Version 2: in-place merge into scores
# =========================================================
def compute_grpo_outcome_advantage_v2(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores



# =========================================================
# Unified test harness
# =========================================================
if __name__ == "__main__":
    # =========================================================
    # Large-scale stress test
    # =========================================================
    torch.manual_seed(42)

    # -------------------------------
    # Large-scale randomized test
    # -------------------------------
    bs = 128
    resp_len = 1

    token_level_rewards = torch.randn(bs, resp_len)
    token_level_rewards = token_level_rewards ** 2
    response_mask = (torch.rand(bs, resp_len) > 0.1).float()

        # -------------------------------
    # Tree-structured data generation
    # -------------------------------
    # current_index: node id (0 .. bs-1)
    current_index = np.arange(bs)

    # depth in {1,2}, depth=1 are roots
    depth = np.random.randint(1, 3, size=bs)

    # index: parent id; roots point to themselves
    index = np.empty(bs, dtype=np.int64)
    for i in range(bs):
        if depth[i] == 1:
            index[i] = np.random.randint(-3,-1)
        else:
            # choose a random root as parent
            candidates = np.where(depth == (depth[i] - 1))[0]
            index[i] = np.random.choice(candidates)

    # index = [3, 3, 1, -3, 0]
    # current_index = [0, 1, 2, 3, 4]
    # depth = [2, 2, 3, 1, 3]
    config = AlgoConfig({
        "grpo_child_score_merge_fn": "mean",
        "merge_weight": 1.0,
        "qe_weight": 1.0,
        "leaf_score_only": False,
    })

    # case 1. 
    index = [-1, -1, -1, 1, 1, 1, 2, 2, 2]
    current_index = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    depth = [1, 1, 1, 2, 2, 2, 2, 2, 2]
    token_level_rewards = torch.tensor([
        [0, 0, 5],
        [0, 0, 1],
        [0, 0, 2], 
        [0, 0, 1],
        [0, 0, 1], 
        [0, 0, 1], 
        [0, 0, 4],
        [0, 0, 3],
        [0, 0, 1], 
    ], dtype=float)
    # merge_weight 1.0, qe_weight 0.0 -> adv: [-0.9073, -0.1650,  1.0722,  0.0000,  0.0000,  0.0000,  0.8729,  0.2182, -1.0911]
    # [0.0000, 1.0000, 2.6667, 1.0000, 1.0000, 1.0000, 4.0000, 3.0000, 1.0000]
    # merge_weight 0.0, qe_weight 1.0 -> adv: [ 1.1209, -0.8006, -0.3203,  0.0000,  0.0000,  0.0000,  0.8729,  0.2182, -1.0911]
    # [5., 1., 2., 1., 1., 1., 4., 3., 1.]
    # merge_weight 1.0, qe_weight 1.0 -> adv: [ 0.6757, -1.1488,  0.4730,  0.0000,  0.0000,  0.0000,  0.8729,  0.2182, -1.0911]
    # [5.0000, 2.0000, 4.6667, 1.0000, 1.0000, 1.0000, 4.0000, 3.0000, 1.0000]

    # case 2
    index = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
    current_index = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    depth = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    response_mask = torch.ones_like(token_level_rewards)

    # case 3  
    index = [-1, -1, -1, -1, -1, -2, -2, -2, -1]
    depth=None
    current_index=None
    response_mask = torch.ones_like(token_level_rewards)

    o_v1, _ = compute_grpo_outcome_advantage_v1(
        token_level_rewards.clone(), response_mask, index, current_index=current_index, depth=depth,
        config=config, step=0
    )
    o_v2, _ = compute_grpo_outcome_advantage_v2(
        token_level_rewards.clone(), response_mask, index, current_index=current_index, depth=depth,
        config=config, step=0
    )

    print("=== Large-scale random test (bs=1024) ===")
    print("allclose:", torch.allclose(o_v1, o_v2))
    print("max diff:", (o_v1 - o_v2).abs().max().item())
    # print("o_v1:", o_v1.squeeze(-1).tolist())

