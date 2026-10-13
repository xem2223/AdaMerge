"""AdaMerge on top of the official ToMe machinery.

This module ports AdaMerge's two contributions -- salience-weighted bipartite
matching (Eq. 3) and adaptive per-layer merge counts (Eq. 4) -- onto the
official facebookresearch/ToMe pipeline. Compared to the original AdaMerge
reference implementation, this version inherits from official ToMe:

  * matching metric  : attention keys, head-averaged (k.mean(1)), not raw x
  * merge location   : between attention and MLP inside each block
  * partition        : index parity ([::2] / [1::2]), CLS protected in-place
  * aggregation      : (salience x size)-weighted average via scatter_reduce
                       (collision-safe), with token-size tracking
  * prop. attention  : attn logits + log(size)

What stays from AdaMerge:
  * salience          : column-sum of row-softmaxed affinity, computed on the
                        SAME metric used for matching (keys), min-max normed
  * matching score    : s_a[i] * cos(a_i, b_j)   (Eq. 3)
  * adaptive r        : r_l = floor(r_max * sigmoid(alpha * z_l))  (Eq. 4)
  * merge weight      : salience * size  (Eq. 5 generalized with sizes)

NOTE: layer statistics (mu_l, sigma_l) MUST be recomputed for this pipeline
(the similarity distribution over keys differs from raw features). Use the
--collect_stats mode of evaluate_adamerge_official.py with the same two-pass
protocol as the paper.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Block, VisionTransformer


# ---------------------------------------------------------------------------
# Salience and adaptive-r (AdaMerge Eq. 2-4, computed on the key metric)
# ---------------------------------------------------------------------------

def compute_salience_from_metric(metric: torch.Tensor, has_cls: bool) -> torch.Tensor:
    """Feature-affinity column-sum salience on the matching metric.

    Args:
        metric: (B, N, D) head-averaged keys for the full sequence.
        has_cls: if True, position 0 is CLS; salience is computed over patch
            tokens only and CLS is assigned salience 1.0 (it is protected from
            merging anyway, so its value only matters for aggregation weights,
            in which it never participates).

    Returns:
        (B, N) salience in [0, 1].
    """
    if has_cls:
        patch = metric[:, 1:, :]
    else:
        patch = metric

    affinity = torch.bmm(patch, patch.transpose(1, 2))
    affinity = torch.nan_to_num(affinity, nan=0.0, posinf=1.0, neginf=-1.0)
    affinity = F.softmax(affinity, dim=-1)
    s = affinity.sum(dim=1)  # column-wise sum

    s_min = s.amin(dim=-1, keepdim=True)
    s_max = s.amax(dim=-1, keepdim=True)
    denom = (s_max - s_min).clamp(min=1e-6)
    s = (s - s_min) / denom

    if has_cls:
        cls_s = torch.ones(s.shape[0], 1, device=s.device, dtype=s.dtype)
        s = torch.cat([cls_s, s], dim=1)
    return s


def compute_adaptive_r(
    weighted_max: torch.Tensor,
    mu_l: float,
    sigma_l: float,
    r_max: int,
    alpha: float,
) -> int:
    """AdaMerge Eq. 4 on the batch-mean of per-row max weighted similarity."""
    s_bar = weighted_max.mean().item()
    if not math.isfinite(s_bar):
        return r_max // 2
    sigma_safe = max(float(sigma_l), 1e-4)
    z = (s_bar - float(mu_l)) / sigma_safe
    score = 1.0 / (1.0 + math.exp(-alpha * z))
    return max(0, min(int(r_max * score), r_max))


# ---------------------------------------------------------------------------
# Salience-weighted bipartite matching + (salience x size)-weighted merge
# ---------------------------------------------------------------------------

def salience_weighted_matching(
    metric: torch.Tensor,
    salience: torch.Tensor,
    r: int,
    class_token: bool,
):
    """Official-ToMe-style matching, with scores weighted by source salience.

    Mirrors tome.merge.bipartite_soft_matching: parity split, CLS protection,
    returns a merge closure. The only change is scores = s_a * cosine.
    Also returns the per-row max weighted similarity (for adaptive-r /
    statistics collection) computed over non-CLS rows.
    """
    protected = 1 if class_token else 0
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    with torch.no_grad():
        metric_n = metric / metric.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        a, b = metric_n[..., ::2, :], metric_n[..., 1::2, :]
        s_a = salience[..., ::2]

        scores = s_a.unsqueeze(-1) * (a @ b.transpose(-1, -2))  # Eq. 3

        if class_token:
            scores[..., 0, :] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        if class_token:
            row_max_stat = node_max[..., 1:]  # exclude CLS row from statistics
        else:
            row_max_stat = node_max

        if r <= 0:
            return None, row_max_stat, 0

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]  # keep CLS first

    def merge(x: torch.Tensor, mode: str = "sum") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src_sel = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src_sel, reduce=mode)
        return torch.cat([unm, dst], dim=1)

    return merge, row_max_stat, r


def merge_salience_wavg(
    merge,
    x: torch.Tensor,
    salience: torch.Tensor,
    size: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(salience x size)-weighted merge; generalizes AdaMerge Eq. 5 with sizes.

    Collision-safe: contributions to a shared destination accumulate via
    scatter_reduce(sum), then normalize by the accumulated weight.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    w = (salience.unsqueeze(-1) * size).clamp(min=1e-6)
    x = merge(x * w, mode="sum")
    w_merged = merge(w, mode="sum")
    x = x / w_merged
    size = merge(size, mode="sum")
    return x, size


# ---------------------------------------------------------------------------
# Patched timm modules (attention identical to official ToMe)
# ---------------------------------------------------------------------------

class AdaMergeAttention(Attention):
    """Official ToMeAttention: proportional attention + return mean key."""

    def forward(self, x: torch.Tensor, size: torch.Tensor = None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if size is not None:  # proportional attention
            attn = attn + size.log()[:, None, None, :, 0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, k.mean(1)


class AdaMergeBlock(Block):
    """AdaMerge applied between attention and MLP (official ToMe position)."""

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        info = self._adamerge_info
        attn_size = info["size"] if info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        layer_idx = info["layer_cursor"]
        info["layer_cursor"] += 1

        salience = compute_salience_from_metric(metric, info["class_token"])
        merge, row_max, _probe_r = salience_weighted_matching(
            metric, salience, r=info["r_max"], class_token=info["class_token"]
        )

        if info["collect_stats"] is not None:
            # per-sample S-bar for statistics extraction
            info["collect_stats"][layer_idx].extend(
                row_max.mean(dim=-1).detach().cpu().tolist()
            )

        stats = info["stats"].get(layer_idx, {"mu": 0.5, "sigma": 0.1})
        r_l = compute_adaptive_r(
            row_max, stats["mu"], stats["sigma"], info["r_max"], info["alpha"]
        )
        info["r_log"].append(r_l)

        if r_l > 0:
            merge, _, r_l = salience_weighted_matching(
                metric, salience, r=r_l, class_token=info["class_token"]
            )
            if merge is not None:
                x, info["size"] = merge_salience_wavg(merge, x, salience, info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


def make_adamerge_class(transformer_class):
    class AdaMergeVisionTransformer(transformer_class):
        def forward(self, *args, **kwargs):
            self._adamerge_info["size"] = None
            self._adamerge_info["layer_cursor"] = 0
            self._adamerge_info["r_log"] = []
            return super().forward(*args, **kwargs)

    return AdaMergeVisionTransformer


def apply_adamerge_patch(
    model: VisionTransformer,
    r_max: int = 23,
    alpha: float = 3.0,
    stats: Optional[dict] = None,
    prop_attn: bool = True,
):
    """Patch a timm ViT with AdaMerge-on-official-ToMe. Set stats after
    calibration; defaults to (mu=0.5, sigma=0.1) per layer as in the paper."""
    model.__class__ = make_adamerge_class(model.__class__)
    model._adamerge_info = {
        "r_max": r_max,
        "alpha": alpha,
        "stats": stats if stats is not None else {},
        "size": None,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "layer_cursor": 0,
        "r_log": [],
        "collect_stats": None,
    }
    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = AdaMergeBlock
            module._adamerge_info = model._adamerge_info
        elif isinstance(module, Attention):
            module.__class__ = AdaMergeAttention
    return model