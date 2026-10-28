import os
import argparse
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GCNConv, Linear


# -----------------------
# utils
# -----------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_dense_matrix(path: str, sep: str = "\t") -> np.ndarray:
    try:
        mat = pd.read_csv(path, header=None, sep=sep).values
    except Exception:
        mat = pd.read_csv(path, header=None, sep=None, engine="python").values
    return mat.astype(np.float32)


def normalize_features(mx: np.ndarray) -> np.ndarray:
    """Row-normalization keeps feature scale stable across datasets."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    return np.diag(r_inv).dot(mx).astype(np.float32)


def read_adj_as_edges(adj_path: str) -> np.ndarray:
    """
    支持：
    1) edge list: 两列 (micro_id, drug_id)
    2) adjacency matrix: Nm x Nd 的 0/1
    """
    df = pd.read_csv(adj_path, header=None, sep=None, engine="python")
    if 2 <= df.shape[1] <= 3:
        edges = df.iloc[:, :2].values.astype(np.int64)
        return edges
    mat = df.values.astype(np.int64)
    mi, dj = np.where(mat > 0)
    return np.stack([mi, dj], axis=1).astype(np.int64)


def topk_edges_from_similarity(sim: np.ndarray, k: int, self_loop: bool = False) -> np.ndarray:
    n = sim.shape[0]
    sim = sim.copy()
    if not self_loop:
        np.fill_diagonal(sim, -1e9)
    k = min(k, n - 1) if n > 1 else 0
    if k <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    edges = np.stack([rows, cols], axis=1).astype(np.int64)
    return np.unique(edges, axis=0)


def split_positive_edges(edges_md: np.ndarray, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    论文式：只对正边做拆分（train/val/test）
    负样本不采样，训练时所有 A=0 都当负
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(edges_md.shape[0])
    rng.shuffle(idx)

    n_total = len(idx)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val - n_test

    train_pos = edges_md[idx[:n_train]]
    val_pos = edges_md[idx[n_train:n_train + n_val]]
    test_pos = edges_md[idx[n_train + n_val:]]
    return train_pos, val_pos, test_pos


def build_A_from_edges(pos_edges: np.ndarray, Nm: int, Nd: int) -> torch.Tensor:
    A = torch.zeros((Nm, Nd), dtype=torch.float32)
    if pos_edges.size > 0:
        A[pos_edges[:, 0], pos_edges[:, 1]] = 1.0
    return A


def sample_fixed_negatives(A_full: torch.Tensor, num_negs: int, seed=42) -> np.ndarray:
    """
    用于评估（可选）：从 A==0 里固定抽一批负样本，避免全矩阵评估太大/太不平衡导致波动。
    注意：这只是“评估采样”，训练仍然是 A=0 全负。
    """
    rng = np.random.default_rng(seed)
    # 转换为 CPU numpy 避免显存溢出
    A_np = A_full.cpu().numpy()
    neg_coords = np.argwhere(A_np == 0)
    if len(neg_coords) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    choose = rng.choice(len(neg_coords), size=min(num_negs, len(neg_coords)), replace=False)
    return neg_coords[choose].astype(np.int64)


def pair_bce_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    target = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
    return F.binary_cross_entropy_with_logits(logits, target)


def pair_asl_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor,
                  gamma_pos: float = 1.0, gamma_neg: float = 4.0,
                  clip: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    pos_prob = torch.sigmoid(pos_logits)
    neg_prob = torch.sigmoid(neg_logits)

    pos_loss = torch.log(pos_prob.clamp(min=eps))
    neg_base = 1.0 - neg_prob
    if clip is not None and clip > 0:
        neg_base = torch.clamp(neg_base + clip, max=1.0)
    neg_loss = torch.log(neg_base.clamp(min=eps))

    if gamma_pos > 0:
        pos_loss = pos_loss * torch.pow(1.0 - pos_prob, gamma_pos)
    if gamma_neg > 0:
        neg_loss = neg_loss * torch.pow(neg_prob, gamma_neg)

    return -(pos_loss.mean() + neg_loss.mean()) * 0.5


def sample_self_paced_negatives(logits: torch.Tensor, A: torch.Tensor, num_negs: int,
                                epoch_idx: int, total_epochs: int, seed: int = 42,
                                k_bins: int = 10) -> np.ndarray:
    if num_negs <= 0:
        return np.zeros((0, 2), dtype=np.int64)

    neg_mask = (A < 0.5)
    neg_coords = torch.nonzero(neg_mask, as_tuple=False)
    if neg_coords.numel() == 0:
        return np.zeros((0, 2), dtype=np.int64)

    neg_scores = torch.sigmoid(logits.detach()[neg_mask]).cpu().numpy().astype(np.float64)
    neg_coords_np = neg_coords.cpu().numpy().astype(np.int64)
    num_negs = min(int(num_negs), len(neg_scores))
    if num_negs <= 0:
        return np.zeros((0, 2), dtype=np.int64)

    rng = np.random.default_rng(seed)
    hardness = neg_scores
    if hardness.max() == hardness.min():
        choose = rng.choice(len(hardness), size=num_negs, replace=False)
        return neg_coords_np[choose]

    alpha = np.tan(np.pi * 0.5 * (epoch_idx / max(total_epochs - 1, 1)))
    populations, edges = np.histogram(hardness, bins=k_bins)
    contributions = np.zeros(k_bins, dtype=np.float64)
    bin_masks = []
    for bin_idx in range(k_bins):
        in_bin = (hardness >= edges[bin_idx]) & (hardness < edges[bin_idx + 1])
        if bin_idx == k_bins - 1:
            in_bin = in_bin | (hardness == edges[bin_idx + 1])
        bin_masks.append(in_bin)
        if populations[bin_idx] > 0:
            contributions[bin_idx] = hardness[in_bin].mean()

    bin_weights = 1.0 / (contributions + alpha + 1e-12)
    bin_weights[~np.isfinite(bin_weights)] = 0.0
    target_per_bin = num_negs * bin_weights / max(bin_weights.sum(), 1e-12)
    target_per_bin[populations == 0] = 0.0
    target_per_bin = target_per_bin.astype(np.int64) + (populations > 0).astype(np.int64)

    sample_probs = np.zeros_like(hardness, dtype=np.float64)
    for bin_idx in range(k_bins):
        if populations[bin_idx] > 0:
            sample_probs[bin_masks[bin_idx]] = target_per_bin[bin_idx] / populations[bin_idx]
    sample_probs[~np.isfinite(sample_probs)] = 0.0
    if sample_probs.sum() <= 0:
        choose = rng.choice(len(hardness), size=num_negs, replace=False)
        return neg_coords_np[choose]

    sample_probs /= sample_probs.sum()
    choose = rng.choice(len(hardness), size=num_negs, replace=False, p=sample_probs)
    return neg_coords_np[choose]


# -----------------------
# Build graph (for message passing)
# -----------------------

def build_hetero_graph(dataset_dir: str, topk: int = 20) -> tuple[HeteroData, np.ndarray]:
    adj_path = os.path.join(dataset_dir, "adj.txt")
    mmi_path = os.path.join(dataset_dir, "microbesimilarity.txt")
    ddi_path = os.path.join(dataset_dir, "drugsimilarity.txt")
    memb_path = os.path.join(dataset_dir, "microbes_embeddings.csv")

    assert os.path.exists(adj_path), f"Missing {adj_path}"
    assert os.path.exists(mmi_path), f"Missing {mmi_path}"
    assert os.path.exists(ddi_path), f"Missing {ddi_path}"
    assert os.path.exists(memb_path), f"Missing {memb_path} (dual-view需要microbes_embeddings.csv)"

    mmi = read_dense_matrix(mmi_path, sep="\t")  # Nm x Nm
    ddi = read_dense_matrix(ddi_path, sep="\t")  # Nd x Nd
    mmi = normalize_features(mmi)
    ddi = normalize_features(ddi)
    Nm, Nd = mmi.shape[0], ddi.shape[0]

    df_emb = pd.read_csv(memb_path, index_col=0)
    x_sem_m = df_emb.values.astype(np.float32)  # Nm x 1536
    x_sem_m = normalize_features(x_sem_m)

    # 对齐数量（保守处理）
    if x_sem_m.shape[0] != Nm:
        n = min(Nm, x_sem_m.shape[0])
        mmi = mmi[:n, :n]
        x_sem_m = x_sem_m[:n]
        Nm = n

    edges_md = read_adj_as_edges(adj_path)

    # 修复索引越界：只保留合法索引
    mask_valid = (
        (edges_md[:, 0] >= 0) & (edges_md[:, 0] < Nm) &
        (edges_md[:, 1] >= 0) & (edges_md[:, 1] < Nd)
    )
    edges_md = edges_md[mask_valid]

    edges_mm = topk_edges_from_similarity(mmi, k=topk, self_loop=False)
    edges_dd = topk_edges_from_similarity(ddi, k=topk, self_loop=False)

    data = HeteroData()
    data["micro"].x_sim = torch.from_numpy(mmi)      # Nm x Nm
    data["micro"].x_sem = torch.from_numpy(x_sem_m)  # Nm x 1536
    data["drug"].x_sim = torch.from_numpy(ddi)       # Nd x Nd

    data["micro"].node_id = torch.arange(Nm)
    data["drug"].node_id = torch.arange(Nd)

    # DMI edges（用于消息传递图结构，异质图：micro-drug）
    edge_index_md = torch.tensor(edges_md.T, dtype=torch.long)
    data["micro", "interacts", "drug"].edge_index = edge_index_md
    data["drug", "rev_interacts", "micro"].edge_index = torch.stack(
        [edge_index_md[1], edge_index_md[0]], dim=0
    )

    # MMI / DDI（同质边：micro-micro / drug-drug）
    mm = torch.tensor(edges_mm.T, dtype=torch.long)
    dd = torch.tensor(edges_dd.T, dtype=torch.long)

    data["micro", "similar", "micro"].edge_index = mm
    data["micro", "similar_rev", "micro"].edge_index = torch.stack([mm[1], mm[0]], dim=0)

    data["drug", "similar", "drug"].edge_index = dd
    data["drug", "similar_rev", "drug"].edge_index = torch.stack([dd[1], dd[0]], dim=0)

    return data, edges_md


# -----------------------
# Contrastive loss (InfoNCE)
# -----------------------

def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """Symmetric InfoNCE with in-batch negatives. Assumes one-to-one alignment."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    labels = torch.arange(z1.size(0), device=z1.device)

    logits12 = (z1 @ z2.t()) / tau
    loss12 = F.cross_entropy(logits12, labels)

    logits21 = (z2 @ z1.t()) / tau
    loss21 = F.cross_entropy(logits21, labels)

    return 0.5 * (loss12 + loss21)


def matrix_bce_loss(logits: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    论文式 closed-world：A=0 全负，A=1 正
    极不平衡 -> pos_weight = #neg/#pos
    """
    pos = A.sum()
    neg = A.numel() - pos
    pos_weight = (neg / (pos + 1e-12)).clamp(min=1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn(logits, A)


def matrix_asl_loss(logits: torch.Tensor, A: torch.Tensor,
                    gamma_pos: float = 1.0, gamma_neg: float = 4.0,
                    clip: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    """
    Asymmetric loss for highly-imbalanced link prediction.
    - down-weights easy negatives more aggressively
    - keeps positives relatively stable
    """
    probs = torch.sigmoid(logits)
    pos_term = probs
    neg_term = 1.0 - probs

    if clip is not None and clip > 0:
        neg_term = torch.clamp(neg_term + clip, max=1.0)

    pos_loss = A * torch.log(pos_term.clamp(min=eps))
    neg_loss = (1.0 - A) * torch.log(neg_term.clamp(min=eps))

    if gamma_pos > 0:
        pos_loss = pos_loss * torch.pow(1.0 - pos_term, gamma_pos) * A
    if gamma_neg > 0:
        neg_loss = neg_loss * torch.pow(probs, gamma_neg) * (1.0 - A)

    pos = A.sum()
    neg = A.numel() - pos
    pos_scale = (neg / (pos + eps)).clamp(min=1.0)

    loss = -(pos_scale * pos_loss + neg_loss)
    return loss.mean()


def hard_negative_loss(logits: torch.Tensor, A: torch.Tensor, neg_ratio: float = 3.0) -> torch.Tensor:
    """Extra penalty on top-scoring negatives to reduce false positives."""
    pos_num = int(A.sum().item())
    if pos_num <= 0:
        return logits.new_zeros(())

    neg_logits = logits[A < 0.5]
    if neg_logits.numel() == 0:
        return logits.new_zeros(())

    topk = min(int(pos_num * neg_ratio), int(neg_logits.numel()))
    if topk <= 0:
        return logits.new_zeros(())

    hard_neg = torch.topk(neg_logits, k=topk, largest=True).values
    target = torch.zeros_like(hard_neg)
    return F.binary_cross_entropy_with_logits(hard_neg, target)


def hard_margin_loss(logits: torch.Tensor, A: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Pull hard positives above hard negatives with a lightweight ranking objective."""
    pos_logits = logits[A > 0.5]
    neg_logits = logits[A < 0.5]
    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return logits.new_zeros(())

    topk = min(int(pos_logits.numel()), int(neg_logits.numel()))
    if topk <= 0:
        return logits.new_zeros(())

    hard_pos = torch.topk(pos_logits, k=topk, largest=False).values
    hard_neg = torch.topk(neg_logits, k=topk, largest=True).values
    return F.relu(margin - hard_pos + hard_neg).mean()

def _sn_sp(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> tuple[float, float]:
    pred = (y_score >= threshold).astype(np.int64)
    y = y_true.astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sn = tp / (tp + fn + 1e-12)
    sp = tn / (tn + fp + 1e-12)
    return float(sn), float(sp)

def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> tuple[float, float, float, float, float]:
    pred = (y_score >= threshold).astype(np.int64)
    y = y_true.astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sn = tp / (tp + fn + 1e-12)
    sp = tn / (tn + fp + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    f1 = 2.0 * precision * sn / (precision + sn + 1e-12)
    return float(acc), float(sn), float(precision), float(sp), float(f1)

def _best_threshold(y_true: np.ndarray, y_score: np.ndarray, metric: str = "youden") -> float:
    best_thr = 0.5
    best_stat = -1e9
    uniq = np.unique(np.clip(y_score.astype(np.float64), 0.0, 1.0))
    if uniq.size == 0:
        return best_thr
    if uniq.size == 1:
        candidates = uniq
    else:
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        candidates = np.concatenate(([max(0.0, uniq[0] - 1e-6)], mids, [min(1.0, uniq[-1] + 1e-6)]))
    for thr in candidates:
        if metric == "acc":
            stat, _, _, _, _ = _binary_metrics(y_true, y_score, threshold=float(thr))
        elif metric == "f1":
            _, _, _, _, stat = _binary_metrics(y_true, y_score, threshold=float(thr))
        else:
            sn, sp = _sn_sp(y_true, y_score, threshold=float(thr))
            stat = sn + sp - 1.0
        if stat > best_stat:
            best_stat = stat
            best_thr = float(thr)
    return best_thr

def split_positive_edges_kfold(edges_md: np.ndarray, num_folds: int, fold_idx: int, seed=42):
    if num_folds < 3:
        raise ValueError("num_folds must be >= 3 for train/val/test split")
    if fold_idx < 0 or fold_idx >= num_folds:
        raise ValueError("fold_idx out of range")

    rng = np.random.default_rng(seed)
    idx = np.arange(edges_md.shape[0])
    rng.shuffle(idx)
    folds = np.array_split(idx, num_folds)

    test_fold = fold_idx
    val_fold = (fold_idx + 1) % num_folds
    train_parts = [folds[i] for i in range(num_folds) if i not in (test_fold, val_fold)]
    train_idx = np.concatenate(train_parts) if len(train_parts) > 0 else np.array([], dtype=np.int64)
    val_idx = folds[val_fold]
    test_idx = folds[test_fold]
    return edges_md[train_idx], edges_md[val_idx], edges_md[test_idx]


def set_interaction_edges(data: HeteroData, edges_md: np.ndarray) -> HeteroData:
    edge_index_md = torch.tensor(edges_md.T, dtype=torch.long)
    data["micro", "interacts", "drug"].edge_index = edge_index_md
    data["drug", "rev_interacts", "micro"].edge_index = torch.stack(
        [edge_index_md[1], edge_index_md[0]], dim=0
    )
    return data


# -----------------------
# Dual-view model
# -----------------------

class DualViewCLModelPaper(nn.Module):
    """
    Encoder: 双视角编码（SIM / SEM）
      - Homogeneous View: 只在相似性子图(MM/DD)上运行 GCNConv
      - Heterogeneous View: 在完整异构图上运行 SAGEConv（MD + MM + DD）
    Decoder: 双线性，全矩阵 logits = Zm W Zd^T
    Loss: 全矩阵 BCE（A=0 全负） + λ * CL
    """

    def __init__(self, in_m_sim: int, in_m_sem: int, in_d_sim: int,
                 hidden=256, out=256, dropout=0.2, use_pair_bias: bool = False,
                 fusion_mode: str = "both", fusion_alpha: float = 0.5,
                 use_decoder_mlp: bool = True):
        super().__init__()
        self.dropout = dropout
        self.out = out
        self.use_pair_bias = use_pair_bias
        self.fusion_mode = fusion_mode
        self.fusion_alpha = fusion_alpha
        self.use_decoder_mlp = use_decoder_mlp

        # input projections
        self.m_sim_proj = nn.Sequential(
            Linear(in_m_sim, hidden), nn.ReLU(), nn.Dropout(dropout), Linear(hidden, hidden)
        )
        self.m_sem_proj = nn.Sequential(
            Linear(in_m_sem, hidden), nn.ReLU(), nn.Dropout(dropout), Linear(hidden, hidden)
        )
        self.d_sim_proj = nn.Sequential(
            Linear(in_d_sim, hidden), nn.ReLU(), nn.Dropout(dropout), Linear(hidden, hidden)
        )

        # conv1: 第一层 GNN（定义所有关系的卷积）
        conv1 = {
            # hetero (micro-drug)
            ("micro", "interacts", "drug"): SAGEConv((-1, -1), hidden),
            ("drug", "rev_interacts", "micro"): SAGEConv((-1, -1), hidden),

            # homogeneous similarity subgraphs
            ("micro", "similar", "micro"): GCNConv(-1, hidden),
            ("micro", "similar_rev", "micro"): GCNConv(-1, hidden),

            ("drug", "similar", "drug"): GCNConv(-1, hidden),
            ("drug", "similar_rev", "drug"): GCNConv(-1, hidden),
        }
        self.gnn1 = HeteroConv(conv1, aggr="sum")

        # conv2: 第二层 GNN
        conv2 = {
            ("micro", "interacts", "drug"): SAGEConv((-1, -1), out),
            ("drug", "rev_interacts", "micro"): SAGEConv((-1, -1), out),

            ("micro", "similar", "micro"): GCNConv(-1, out),
            ("micro", "similar_rev", "micro"): GCNConv(-1, out),

            ("drug", "similar", "drug"): GCNConv(-1, out),
            ("drug", "similar_rev", "drug"): GCNConv(-1, out),
        }
        self.gnn2 = HeteroConv(conv2, aggr="sum")

        # CL heads
        self.cl_head_m = nn.Sequential(nn.Linear(out, out), nn.ReLU(), nn.Linear(out, out))
        self.cl_head_d = nn.Sequential(nn.Linear(out, out), nn.ReLU(), nn.Linear(out, out))

        # fuse micro / drug
        self.fuse_m = nn.Sequential(nn.Linear(out * 2, out), nn.ReLU())
        self.fuse_d = nn.Sequential(nn.Linear(out * 2, out), nn.ReLU())

        # bilinear decoder parameter
        self.W = nn.Parameter(torch.empty(out, out))
        nn.init.xavier_uniform_(self.W)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(out * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.decoder_res_weight = nn.Parameter(torch.tensor(0.5))
        self.micro_bias_head = nn.Linear(out, 1)
        self.drug_bias_head = nn.Linear(out, 1)
        self.decoder_bias_weight = nn.Parameter(torch.tensor(0.2))
        self.decode_chunk_size = 128

    # ---------- 两个视角的 message passing ----------

    def _homo_forward(self, data: HeteroData, micro_x: torch.Tensor, drug_x: torch.Tensor):
        """
        同质分支：只传递相似性边 (similar, similar_rev) 的 GCNConv
        """
        x_dict = {"micro": micro_x, "drug": drug_x}

        # 只取相似性边
        edge_index_dict = {
            k: v for k, v in data.edge_index_dict.items()
            if "similar" in k[1]
        }

        x_dict = self.gnn1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}
        x_dict = self.gnn2(x_dict, edge_index_dict)
        return x_dict

    def _hetero_forward(self, data: HeteroData, micro_x: torch.Tensor, drug_x: torch.Tensor):
        """
        异质分支：使用整张异构图（MD + MM + DD）
        - MM/DD 边用 GCNConv
        - MD 边用 SAGEConv
        """
        x_dict = {"micro": micro_x, "drug": drug_x}

        # 这里直接用 data.edge_index_dict，包含：
        #   ("micro","interacts","drug"),
        #   ("drug","rev_interacts","micro"),
        #   ("micro","similar","micro"),
        #   ("micro","similar_rev","micro"),
        #   ("drug","similar","drug"),
        #   ("drug","similar_rev","drug")
        edge_index_dict = data.edge_index_dict

        x_dict = self.gnn1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}
        x_dict = self.gnn2(x_dict, edge_index_dict)
        return x_dict

    def encode_views(self, data: HeteroData):
        """
        Strict Dual-View:
        1. Homogeneous View (SIM): micro_sim + drug_sim -> GCN on similarity graph (MM/DD)
        2. Heterogeneous View (SEM): micro_sem + drug_sim -> SAGE+GCN on full hetero graph (MD+MM+DD)
        """
        m_sim = self.m_sim_proj(data["micro"].x_sim)
        d_sim = self.d_sim_proj(data["drug"].x_sim)

        m_sem = self.m_sem_proj(data["micro"].x_sem)
        d_sem = d_sim  # Drug 没有语义特征，复用 sim 特征

        # View 1: 同质视角
        z_sim = self._homo_forward(data, m_sim, d_sim)

        # View 2: 异质视角（完整异构图）
        z_sem = self._hetero_forward(data, m_sem, d_sem)

        return z_sim, z_sem

    # ---------- 融合 & 解码 ----------

    def fuse(self, z_sim, z_sem):
        if self.fusion_mode == "sim_only":
            z_m = z_sim["micro"]
            z_d = z_sim["drug"]
        elif self.fusion_mode == "sem_only":
            z_m = z_sem["micro"]
            z_d = z_sem["drug"]
        elif self.fusion_mode == "weighted":
            alpha = float(self.fusion_alpha)
            z_m = alpha * z_sim["micro"] + (1.0 - alpha) * z_sem["micro"]
            z_d = alpha * z_sim["drug"] + (1.0 - alpha) * z_sem["drug"]
        elif self.fusion_mode == "mean":
            z_m = 0.5 * (z_sim["micro"] + z_sem["micro"])
            z_d = 0.5 * (z_sim["drug"] + z_sem["drug"])
        else:
            z_m = self.fuse_m(torch.cat([z_sim["micro"], z_sem["micro"]], dim=-1))
            z_d = self.fuse_d(torch.cat([z_sim["drug"], z_sem["drug"]], dim=-1))
        return {"micro": z_m, "drug": z_d}

    def decode_all(self, z_fused):
        """
        全矩阵 logits: [Nm, Nd]
        """
        Zm = z_fused["micro"]  # [Nm, d]
        Zd = z_fused["drug"]   # [Nd, d]
        bilinear_logits = (Zm @ self.W) @ Zd.t()

        logits = bilinear_logits
        if self.use_decoder_mlp:
            mlp_chunks = []
            for start in range(0, Zm.size(0), self.decode_chunk_size):
                end = min(start + self.decode_chunk_size, Zm.size(0))
                zm_chunk = Zm[start:end]
                expand_m = zm_chunk.unsqueeze(1).expand(-1, Zd.size(0), -1)
                expand_d = Zd.unsqueeze(0).expand(zm_chunk.size(0), -1, -1)
                pair_feat = torch.cat(
                    [expand_m, expand_d, expand_m * expand_d, torch.abs(expand_m - expand_d)],
                    dim=-1,
                )
                mlp_chunks.append(self.decoder_mlp(pair_feat).squeeze(-1))

            mlp_logits = torch.cat(mlp_chunks, dim=0)
            logits = logits + self.decoder_res_weight * mlp_logits
        if self.use_pair_bias:
            micro_bias = self.micro_bias_head(Zm)
            drug_bias = self.drug_bias_head(Zd).t()
            logits = logits + self.decoder_bias_weight * (micro_bias + drug_bias)
        return logits

    @torch.no_grad()
    def export_embeddings(self, data: HeteroData):
        was_training = self.training
        self.eval()
        z_sim, z_sem = self.encode_views(data)
        z_fused = self.fuse(z_sim, z_sem)
        out = {
            "micro_raw": data["micro"].x_sem.detach().cpu(),
            "drug_raw": data["drug"].x_sim.detach().cpu(),
            "micro_sim": z_sim["micro"].detach().cpu(),
            "drug_sim": z_sim["drug"].detach().cpu(),
            "micro_sem": z_sem["micro"].detach().cpu(),
            "drug_sem": z_sem["drug"].detach().cpu(),
            "micro_fused": z_fused["micro"].detach().cpu(),
            "drug_fused": z_fused["drug"].detach().cpu(),
        }
        if was_training:
            self.train()
        return out

    def cl_loss(self, z_sim, z_sem, cl_on_drug: bool, tau: float):
        # micro: sim vs sem
        m1 = self.cl_head_m(z_sim["micro"])
        m2 = self.cl_head_m(z_sem["micro"])
        loss_m = info_nce(m1, m2, tau=tau)

        if not cl_on_drug:
            return loss_m, loss_m.detach() * 0.0

        # drug: sim vs sem
        d1 = F.dropout(z_sim["drug"], p=self.dropout, training=self.training)
        d2 = F.dropout(z_sem["drug"], p=self.dropout, training=self.training)
        d1 = self.cl_head_d(d1)
        d2 = self.cl_head_d(d2)
        loss_d = info_nce(d1, d2, tau=tau)
        return loss_m, loss_d


# -----------------------
# train / eval
# -----------------------

def train_one_epoch_paper(model, data, A_train, opt, device,
                          lam_cl=0.0, tau=0.2, cl_on_drug=False,
                          lam_hard=0.0, hard_neg_ratio=3.0,
                          lam_rank=0.0, rank_margin=0.5, grad_clip=1.0,
                          pred_loss: str = "bce",
                          asl_gamma_pos: float = 1.0,
                          asl_gamma_neg: float = 4.0,
                          asl_clip: float = 0.05,
                          train_pos_edges: Optional[np.ndarray] = None,
                          train_neg_edges: Optional[np.ndarray] = None,
                          train_neg_strategy: str = "random",
                          epoch_idx: int = 0,
                          total_epochs: int = 1,
                          self_paced_bins: int = 10,
                          sample_seed: int = 42):
    model.train()
    data = data.to(device)
    A_train = A_train.to(device)

    opt.zero_grad()

    z_sim, z_sem = model.encode_views(data)
    z_fused = model.fuse(z_sim, z_sem)

    logits = model.decode_all(z_fused)  # [Nm,Nd]
    if train_pos_edges is not None and train_neg_edges is None and train_neg_strategy == "self_paced":
        train_neg_edges = sample_self_paced_negatives(
            logits=logits,
            A=A_train,
            num_negs=len(train_pos_edges),
            epoch_idx=epoch_idx,
            total_epochs=total_epochs,
            seed=sample_seed,
            k_bins=self_paced_bins,
        )
    if train_pos_edges is not None and train_neg_edges is not None:
        pos_idx = torch.as_tensor(train_pos_edges, dtype=torch.long, device=device)
        neg_idx = torch.as_tensor(train_neg_edges, dtype=torch.long, device=device)
        pos_logits = logits[pos_idx[:, 0], pos_idx[:, 1]]
        neg_logits = logits[neg_idx[:, 0], neg_idx[:, 1]]
        if pred_loss == "asl":
            loss_pred = pair_asl_loss(
                pos_logits, neg_logits,
                gamma_pos=asl_gamma_pos,
                gamma_neg=asl_gamma_neg,
                clip=asl_clip,
            )
        else:
            loss_pred = pair_bce_loss(pos_logits, neg_logits)
    else:
        if pred_loss == "asl":
            loss_pred = matrix_asl_loss(
                logits, A_train,
                gamma_pos=asl_gamma_pos,
                gamma_neg=asl_gamma_neg,
                clip=asl_clip,
            )
        else:
            loss_pred = matrix_bce_loss(logits, A_train)
    loss_hard = hard_negative_loss(logits, A_train, neg_ratio=hard_neg_ratio)
    loss_rank = hard_margin_loss(logits, A_train, margin=rank_margin)

    loss_m, loss_d = model.cl_loss(z_sim, z_sem,
                                   cl_on_drug=cl_on_drug, tau=tau)
    loss = loss_pred + lam_cl * (loss_m + loss_d) + lam_hard * loss_hard + lam_rank * loss_rank

    loss.backward()
    if grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    opt.step()

    return (
        float(loss.item()),
        float(loss_pred.item()),
        float(loss_m.item()),
        float(loss_d.item()),
        float(loss_hard.item()),
        float(loss_rank.item()),
    )


@torch.no_grad()
def eval_fixed_pos_neg(model, data, pos_edges: np.ndarray,
                       neg_edges: np.ndarray, device,
                       threshold: float = 0.5,
                       tune_threshold: bool = False,
                       threshold_metric: str = "youden"):
    """
    评估：用固定的 pos/test + 固定采样的 neg（一次抽定）算 AUC/AP
    训练仍是 closed-world 全负
    """
    model.eval()
    data = data.to(device)

    z_sim, z_sem = model.encode_views(data)
    z_fused = model.fuse(z_sim, z_sem)
    logits_all = model.decode_all(z_fused)  # [Nm,Nd]

    pos_scores = torch.sigmoid(
        logits_all[pos_edges[:, 0], pos_edges[:, 1]]
    ).cpu().numpy()
    neg_scores = torch.sigmoid(
        logits_all[neg_edges[:, 0], neg_edges[:, 1]]
    ).cpu().numpy()

    y = np.concatenate([np.ones_like(pos_scores),
                        np.zeros_like(neg_scores)])
    s = np.concatenate([pos_scores, neg_scores])

    auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else 0.5
    aupr = average_precision_score(y, s)
    if tune_threshold:
        threshold = _best_threshold(y, s, metric=threshold_metric)
    acc, sn, precision, sp, f1 = _binary_metrics(y, s, threshold=threshold)
    return auc, aupr, acc, sn, precision, sp, f1, float(threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="./dataset/MDAD")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=200)

    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    # 论文式：训练不采样负边；评估可固定采样一批负边（推荐）
    ap.add_argument(
        "--eval_neg_ratio",
        type=float,
        default=10.0,
        help="评估时负样本数量 = eval_neg_ratio * #pos_test（只用于评估，不影响训练）"
    )

    # CL params（可选）
    ap.add_argument("--lam_cl", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--cl_on_drug", type=int, default=0)
    ap.add_argument("--lam_hard", type=float, default=0.2)
    ap.add_argument("--hard_neg_ratio", type=float, default=3.0)
    ap.add_argument("--lam_rank", type=float, default=0.0)
    ap.add_argument("--rank_margin", type=float, default=0.5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--pred_loss", type=str, default="bce", choices=["bce", "asl"])
    ap.add_argument("--asl_gamma_pos", type=float, default=1.0)
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--num_folds", type=int, default=1, help=">=3 to enable k-fold train/val/test split")
    ap.add_argument("--fold_idx", type=int, default=0, help="current fold index when num_folds>=3")
    ap.add_argument("--train_graph_only", type=int, default=0, help="use only train_pos edges for micro-drug message passing")
    ap.add_argument("--threshold_metric", type=str, default="youden", choices=["youden", "acc", "f1"])
    ap.add_argument("--train_neg_ratio", type=float, default=0.0, help=">0 enables sampled train negatives: #neg = ratio * #train_pos")
    ap.add_argument("--train_neg_strategy", type=str, default="random", choices=["random", "self_paced"])
    ap.add_argument("--self_paced_bins", type=int, default=10)
    ap.add_argument("--use_pair_bias", type=int, default=0, help="enable lightweight row/col decoder bias head")
    ap.add_argument("--fusion_mode", type=str, default="both", choices=["both", "sim_only", "sem_only", "weighted", "mean"])
    ap.add_argument("--fusion_alpha", type=float, default=0.5, help="used when fusion_mode=weighted")
    ap.add_argument("--use_decoder_mlp", type=int, default=1, help="enable lightweight residual pair decoder")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, edges_md = build_hetero_graph(args.dataset_dir, topk=args.topk)
    Nm = data["micro"].x_sim.size(0)
    Nd = data["drug"].x_sim.size(0)

    # 只拆分正边
    if args.num_folds >= 3:
        train_pos, val_pos, test_pos = split_positive_edges_kfold(
            edges_md,
            num_folds=args.num_folds,
            fold_idx=args.fold_idx,
            seed=args.seed,
        )
    else:
        train_pos, val_pos, test_pos = split_positive_edges(
            edges_md,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    # 训练标签矩阵：只包含 train_pos 为 1，其它全 0
    A_train = build_A_from_edges(train_pos, Nm, Nd)
    A_full = build_A_from_edges(edges_md, Nm, Nd)

    if bool(args.train_graph_only):
        data = set_interaction_edges(data, train_pos)

    # 固定采样评估负样本（一次抽定）
    num_test_negs = int(len(test_pos) * args.eval_neg_ratio)
    num_val_negs = int(len(val_pos) * args.eval_neg_ratio)

    val_neg = sample_fixed_negatives(A_full,
                                     num_negs=num_val_negs,
                                     seed=args.seed + 1)
    test_neg = sample_fixed_negatives(A_full,
                                      num_negs=num_test_negs,
                                      seed=args.seed + 2)

    # model
    in_m_sim = data["micro"].x_sim.size(-1)
    in_m_sem = data["micro"].x_sem.size(-1)
    in_d_sim = data["drug"].x_sim.size(-1)

    model = DualViewCLModelPaper(
        in_m_sim=in_m_sim,
        in_m_sem=in_m_sem,
        in_d_sim=in_d_sim,
        hidden=args.hidden,
        out=args.out,
        dropout=args.dropout,
        use_pair_bias=bool(args.use_pair_bias),
        fusion_mode=args.fusion_mode,
        fusion_alpha=args.fusion_alpha,
        use_decoder_mlp=bool(args.use_decoder_mlp),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=20, min_lr=1e-5
    )

    best_val = -1.0
    best_state = None
    no_improve_epochs = 0

    for epoch in range(1, args.epochs + 1):
        loss, lp, lm, ld, lh, lrk = train_one_epoch_paper(
            model,
            data,
            A_train,
            opt,
            device,
            lam_cl=args.lam_cl,
            tau=args.tau,
            cl_on_drug=bool(args.cl_on_drug),
            lam_hard=args.lam_hard,
            hard_neg_ratio=args.hard_neg_ratio,
            lam_rank=args.lam_rank,
            rank_margin=args.rank_margin,
            grad_clip=args.grad_clip,
            pred_loss=args.pred_loss,
            asl_gamma_pos=args.asl_gamma_pos,
            asl_gamma_neg=args.asl_gamma_neg,
            asl_clip=args.asl_clip,
            train_pos_edges=train_pos if args.train_neg_ratio > 0 else None,
            train_neg_edges=sample_fixed_negatives(
                A_full,
                num_negs=int(len(train_pos) * args.train_neg_ratio),
                seed=args.seed + 1000 + epoch,
            ) if (args.train_neg_ratio > 0 and args.train_neg_strategy == "random") else None,
            train_neg_strategy=args.train_neg_strategy,
            epoch_idx=epoch - 1,
            total_epochs=args.epochs,
            self_paced_bins=args.self_paced_bins,
            sample_seed=args.seed + 1000 + epoch,
        )

        val_auc, val_ap, val_acc, val_sn, val_pr, val_sp, val_f1, best_thr = eval_fixed_pos_neg(
            model, data, val_pos, val_neg, device, tune_threshold=True,
            threshold_metric=args.threshold_metric
        )
        test_auc, test_ap, test_acc, test_sn, test_pr, test_sp, test_f1, _ = eval_fixed_pos_neg(
            model, data, test_pos, test_neg, device, threshold=best_thr
        )

        if val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        scheduler.step(val_auc)

        if epoch == 1 or epoch % 10 == 0:
            current_lr = opt.param_groups[0]["lr"]
            print(
                f"[{epoch:03d}] loss={loss:.4f} "
                f"(pred={lp:.4f}, hard={lh:.4f}, rank={lrk:.4f}, cl_m={lm:.4f}, cl_d={ld:.4f}) | "
                f"lr={current_lr:.2e} | "
                f"val AUC={val_auc:.4f} AUPR={val_ap:.4f} ACC={val_acc:.4f} SN={val_sn:.4f} PR={val_pr:.4f} SP={val_sp:.4f} F1={val_f1:.4f} thr={best_thr:.2f} | "
                f"test AUC={test_auc:.4f} AUPR={test_ap:.4f} ACC={test_acc:.4f} SN={test_sn:.4f} PR={test_pr:.4f} SP={test_sp:.4f} F1={test_f1:.4f}"
            )

        if no_improve_epochs >= args.patience and (val_auc + args.min_delta) < best_val:
            print(f"Early stop at epoch {epoch:03d}, best val AUC={best_val:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_auc, val_ap, val_acc, val_sn, val_pr, val_sp, val_f1, best_thr = eval_fixed_pos_neg(
        model, data, val_pos, val_neg, device, tune_threshold=True,
        threshold_metric=args.threshold_metric
    )
    final_auc, final_ap, final_acc, final_sn, final_pr, final_sp, final_f1, _ = eval_fixed_pos_neg(
        model, data, test_pos, test_neg, device, threshold=best_thr
    )
    print(f"\nBest Val AUC={best_val:.4f}")
    print(
        f"Final Test AUC={final_auc:.4f} AUPR={final_ap:.4f} "
        f"ACC={final_acc:.4f} SN={final_sn:.4f} PR={final_pr:.4f} "
        f"SP={final_sp:.4f} F1={final_f1:.4f} thr={best_thr:.2f}"
    )

    save_path = os.path.join(args.dataset_dir,
                             "paperstyle_closedworld_best.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
