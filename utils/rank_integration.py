import torch
import torch.nn as nn


class DifferentiableRankIntegration(nn.Module):
    """Differentiable Rank Integration (DRI) for multi-modal fusion.

    Replaces a simple weighted-sum of similarity matrices with a rank-aware
    consensus score.  The module has **no learnable parameters**; tau and k
    are fixed hyperparameters.

    Pipeline:
        1. Expert Soft Ranks   – convert per-expert cosine-similarity matrices
           into differentiable rank estimates via sigmoid counting.
        2. Fused Consensus Score – aggregate the two rank matrices with
           per-pair modality weights using a reciprocal-rank formula.

    Args:
        tau:  Temperature for the sigmoid in the soft-rank computation.
              Lower values make the rank "harder" (closer to integer ranks).
              Used when ``dynamic_tau=False``.
        k:    Smoothing constant in the reciprocal-rank formula.
              Larger k dampens rank differences.
        dynamic_tau: When True, tau is computed per-modality per-batch as
                     max(std(sim), 0.01) instead of using the fixed value.
        chunk_size: Number of query rows processed at once to bound peak
                    GPU memory (avoids a full [B, B, B] allocation).
    """

    def __init__(self, tau: float = 0.1, k: float = 60.0,
                 dynamic_tau: bool = False, chunk_size: int = 64):
        super().__init__()
        self.tau = tau
        self.k = k
        self.dynamic_tau = dynamic_tau
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Expert Soft Ranks
    # ------------------------------------------------------------------
    def compute_expert_soft_ranks(
        self,
        sim: torch.Tensor,
        pos_mask: torch.BoolTensor,
        neg_mask: torch.BoolTensor,
        tau: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute differentiable soft ranks for one expert modality.

        For a positive pair (i, j):
            r_hat_{ij} = 1 + sum_{k in N_i} sigma((s_ik - s_ij) / tau)

        For a negative pair (i, j):
            r_hat_{ij} = 1 + sum_{m in P_i} sigma((s_im - s_ij) / tau)

        Uses row-chunked broadcasting so peak memory is
        [chunk_size, B, B] instead of [B, B, B].

        Args:
            sim:      [B, B] cosine-similarity matrix for the expert.
            pos_mask: [B, B] bool – True where (i, j) is a positive pair.
            neg_mask: [B, B] bool – True where (i, j) is a negative pair.
            tau:      Temperature override.  When *None* falls back to
                      ``self.tau`` (the fixed hyperparameter).

        Returns:
            rank: [B, B] soft-rank matrix.  Entries where neither mask is
                  True are set to 1 (unused by downstream loss).
        """
        if tau is None:
            tau = self.tau

        B = sim.size(0)
        rank = torch.ones_like(sim)

        pos_f = pos_mask.float()
        neg_f = neg_mask.float()

        for start in range(0, B, self.chunk_size):
            end = min(start + self.chunk_size, B)

            chunk_sim = sim[start:end]                                  # [C, B]
            # diff[c, k, j] = sim[i_c, k] - sim[i_c, j]
            diff = chunk_sim.unsqueeze(2) - chunk_sim.unsqueeze(1)      # [C, B_k, B_j]
            sig = torch.sigmoid(diff / tau)                             # [C, B_k, B_j]

            # Positive-pair rank: count negatives that beat this positive
            neg_k = neg_f[start:end].unsqueeze(2)                       # [C, B_k, 1]
            rank_pos = 1.0 + (sig * neg_k).sum(dim=1)                  # [C, B_j]

            # Negative-pair rank: count positives that beat this negative
            pos_k = pos_f[start:end].unsqueeze(2)                       # [C, B_k, 1]
            rank_neg = 1.0 + (sig * pos_k).sum(dim=1)                  # [C, B_j]

            rank[start:end] = (
                rank_pos * pos_f[start:end]
                + rank_neg * neg_f[start:end]
            )

        return rank

    # ------------------------------------------------------------------
    # Fused Consensus Score
    # ------------------------------------------------------------------
    def forward(
        self,
        s_v: torch.Tensor,
        s_l: torch.Tensor,
        pos_mask: torch.BoolTensor,
        neg_mask: torch.BoolTensor,
        w_v: torch.Tensor,
        w_l: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the fused consensus score S_hat.

        S_hat_{ij} = (k + 1) * [ w_v_{ij} / (k + r_hat_v_{ij})
                                + w_l_{ij} / (k + r_hat_l_{ij}) ]

        Dynamic tau: instead of a single fixed temperature, each
        modality's tau is set to the batch standard deviation of its
        similarity matrix (floored at 0.01) and detached so that no
        gradient flows through the std computation.

        Args:
            s_v:      [B, B] visual cosine-similarity matrix.
            s_l:      [B, B] textual cosine-similarity matrix.
            pos_mask: [B, B] bool – True for positive pairs.
            neg_mask: [B, B] bool – True for negative pairs.
            w_v:      [B, B] pairwise visual modality weights.
            w_l:      [B, B] pairwise textual modality weights.

        Returns:
            S_hat: [B, B] fused consensus score matrix, suitable as a
                   drop-in replacement for cosine similarity in MS loss.
        """
        if self.dynamic_tau:
            tau_v = torch.clamp(s_v.std(), min=0.01).detach()
            tau_l = torch.clamp(s_l.std(), min=0.01).detach()
            self.last_tau_v = tau_v
            self.last_tau_l = tau_l
        elif hasattr(self, 'precomputed_tau_v'):
            tau_v = self.precomputed_tau_v
            tau_l = self.precomputed_tau_l
            self.last_tau_v = tau_v
            self.last_tau_l = tau_l
        else:
            tau_v = self.tau
            tau_l = self.tau

        rank_v = self.compute_expert_soft_ranks(s_v, pos_mask, neg_mask, tau=tau_v)
        rank_l = self.compute_expert_soft_ranks(s_l, pos_mask, neg_mask, tau=tau_l)

        S_hat = (self.k + 1) * (
            w_v / (self.k + rank_v) + w_l / (self.k + rank_l)
        )
        return S_hat
