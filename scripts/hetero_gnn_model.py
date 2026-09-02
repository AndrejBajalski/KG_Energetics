"""
hetero_gnn.py

Heterogeneous multi-task GNN for the KG_Energetics feeder: predicts bus voltage
(node-level) and line current (edge-level) from the graph snapshots produced by
FeederHeteroSnapshotDataset (see pyg_dataset.py).

Run this file directly to train:  python hetero_gnn.py

Requires: torch, torch_geometric, neo4j, pandas
"""

import copy
import csv
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, SAGEConv, GATv2Conv

from hetero_gnn_dataset import FeederHeteroSnapshotDataset, BASE_DIR


class FeederMultiTaskGNN(torch.nn.Module):
    def __init__(self, metadata, in_dims, edge_attr_dim, hidden=64):
        super().__init__()
        self.lin_in = torch.nn.ModuleDict({
            ntype: torch.nn.Linear(dim, hidden) for ntype, dim in in_dims.items()
        })
        # Only the line_segment relations carry edge_attr (length, phase
        # one-hot, and r1/x1/r0/x0/c1/c0 impedance). SAGEConv IGNORES edge
        # attributes, so with it the impedance, length and per-line phases
        # never entered any node embedding — they reached only the current
        # head, which concatenates edge_attr explicitly. That left the
        # voltage head structurally unable to see the quantities voltage
        # drop actually depends on. GATv2Conv consumes edge_attr (edge_dim),
        # so on the line_segment / rev_line_segment relations the edge
        # features now flow into the bus embeddings and reach the voltage
        # head. Relations with no edge_attr keep plain SAGEConv.
        def _hetero(metadata, hidden, edge_attr_dim):
            convs = {}
            for rel in metadata[1]:
                if "line_segment" in rel[1]:
                    convs[rel] = GATv2Conv(
                        (-1, -1), hidden, edge_dim=edge_attr_dim,
                        add_self_loops=False,
                    )
                else:
                    convs[rel] = SAGEConv((-1, -1), hidden)
            return HeteroConv(convs, aggr="sum")

        self.conv1 = _hetero(metadata, hidden, edge_attr_dim)
        self.conv2 = _hetero(metadata, hidden, edge_attr_dim)

        # Node task output (Voltage) — an MLP, not a bare Linear. A single
        # Linear can only form weighted sums of the embedding, but the
        # dominant voltage-drop term is a PRODUCT (path impedance x demand).
        # A hidden ReLU layer gives the head the capacity to approximate
        # that product from the two per-bus ingredients now in the
        # embedding (cumulative impedance + downstream demand).
        self.out_v = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

        # Edge task output (Current) -> expects concatenated (src_emb, dst_emb, edge_attr)
        self.out_i = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2 + edge_attr_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1)
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Input projection
        h_dict = {k: F.relu(self.lin_in[k](v)) for k, v in x_dict.items()}

        # edge_attr_dict only carries keys for the line_segment relations;
        # HeteroConv routes it per-relation, so the GATv2 convs receive it
        # and the SAGEConv convs (no matching key) are called without it.
        # Message Passing Layer 1
        out1 = self.conv1(h_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        h_dict = {**h_dict, **out1}
        h_dict = {k: F.relu(v) for k, v in h_dict.items()}

        # Message Passing Layer 2
        out2 = self.conv2(h_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        h_dict = {**h_dict, **out2}

        # Predict Bus Voltage
        pred_v = self.out_v(h_dict["bus"]).squeeze(-1)

        # Predict Line Current
        line_rel = ("bus", "line_segment", "bus")
        if line_rel in edge_index_dict:
            src, dst = edge_index_dict[line_rel]
            src_emb = h_dict["bus"][src]
            dst_emb = h_dict["bus"][dst]
            edge_attr = edge_attr_dict[line_rel]

            edge_features = torch.cat([src_emb, dst_emb, edge_attr], dim=-1)
            pred_i = self.out_i(edge_features).squeeze(-1)
        else:
            pred_i = None

        return pred_v, pred_i


if __name__ == "__main__":
    LAMBDA_CURRENT = 1.0

    # Early stopping. Validation current is noisy (it bounces epoch-to-epoch
    # because the held-out split is a later part of the day), so we stop on
    # PATIENCE rather than the first non-improvement, and we keep the BEST
    # epoch's weights (in memory) instead of whatever the final epoch lands on.
    # MAX_EPOCHS is a ceiling; PATIENCE is how many epochs without a new best
    # combined-val score we tolerate before stopping.
    MAX_EPOCHS = 20
    PATIENCE = 10
    # Ablation: UPSTREAM_OF gives every bus a direct edge to every
    # downstream bus. Current is very sensitive to real-time downstream
    # demand, and with only 2 SAGEConv layers a line more than 2 hops from
    # the loads it serves never receives that signal in time — this is a
    # cheap way to test whether that's a limiting factor. Toggle to compare
    # against the default (False).
    INCLUDE_UPSTREAM = False

    # Per-epoch train/validation metrics are appended here and flushed as soon
    # as they are computed, so an interrupted or crashed run still leaves a
    # usable partial log rather than nothing. NOTE: on Windows a leading "/"
    # is drive-relative, so "/temp/..." resolves to <current drive>:\temp\,
    # e.g. C:\temp\hetero_gnn_results.csv. The directory is created if absent.
    RESULTS_CSV = Path("/temp/hetero_gnn_results.csv")
    # 'eval_*' holds whichever split the row's 'split' column names, so the same
    # columns carry validation during training and test in the final rows.
    RESULTS_CSV_FIELDS = [
        "split", "epoch",
        "train_mse_v_vpu2", "train_mse_i_a2",
        "eval_mse_v_vpu2", "eval_mse_i_a2",
        "eval_mse_v_norm", "eval_mse_i_norm",
        "eval_r2_v", "eval_r2_i", "eval_combined",
        "is_best",
    ]

    ds = FeederHeteroSnapshotDataset(
        neo4j_uri=os.environ.get("NEO4J_URI"),
        neo4j_user=os.environ.get("NEO4J_USERNAME"),
        neo4j_password=os.environ.get("NEO4J_PASSWORD"),
        load_profiles_path=BASE_DIR / "data" / "Computed" / "load_profiles.csv",
        voltage_labels_path=BASE_DIR / "data" / "Computed" / "bus_voltages.csv",
        current_labels_path=BASE_DIR / "data" / "Computed" / "line_currents.csv",
        include_upstream=INCLUDE_UPSTREAM,
    )
    train_ds, val_ds, test_ds = ds.time_split()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    sample = ds[0]
    in_dims = {ntype: sample[ntype].x.size(1) for ntype in sample.node_types}
    line_rel = ("bus", "line_segment", "bus")
    edge_attr_dim = sample[line_rel].edge_attr.size(1)

    model = FeederMultiTaskGNN(sample.metadata(), in_dims, edge_attr_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Per-line current target stats, fit on the training split inside
    # time_split() -> fit_normalization(). Used to un-normalize predictions
    # back to real Amps for logging. The graph topology (and therefore line
    # order) is identical across every snapshot, so repeating this vector
    # by the number of graphs in a batch lines up with the batched edge order.
    i_mean, i_std = ds.norm_stats[("current_target",)]
    # Same for the per-bus voltage target stats. Both heads now train on
    # normalized targets and are un-normalized back to real units (Vpu /
    # Amps) purely for logging and R^2, so the reported MSE stays comparable
    # to the trivial baselines (~0.000042 Vpu^2, ~47 A^2 train).
    v_mean, v_std = ds.norm_stats[("voltage_target",)]

    def evaluate(loader):
        model.eval()
        total_v, total_v_norm, total_i_real, total_i_norm, n = 0.0, 0.0, 0.0, 0.0, 0

        # --- Per-entity R^2 accumulation ---------------------------------
        # R^2 = 1 - SS_res / SS_tot, where SS_tot is measured against each
        # entity's OWN mean (per-bus for voltage, per-line for current), not
        # a single global mean. Using each entity's own variance is what
        # makes this a fair "explained variance" score: a global mean would
        # credit the model for merely knowing that a trunk line carries far
        # more current than a lateral, inflating R^2. This matches
        # sklearn.r2_score(..., multioutput="variance_weighted").
        #
        # Both heads are scored in REAL units (Vpu, Amps). Each snapshot
        # carries the same entity set in the same order, so reshaping a
        # batch's flat predictions to [num_graphs, num_entities] recovers the
        # per-entity columns; we accumulate, per entity, sum(y), sum(y^2),
        # sum(residual^2) and a count of snapshots, then combine at the end:
        #   SS_tot = sum_e[ sum(y^2) - sum(y)^2 / count ]   (per-entity centered)
        # Unlabeled entities (NaN target) are zeroed and dropped via the mask.
        acc = {
            "v": {"sy": None, "sy2": 0.0, "sres": 0.0, "graphs": 0, "valid": None},
            "i": {"sy": None, "sy2": 0.0, "sres": 0.0, "graphs": 0, "valid": None},
        }

        def _accumulate(a, y_flat, p_flat, num_graphs):
            n_ent = y_flat.numel() // num_graphs
            y_r = y_flat.view(num_graphs, n_ent)
            p_r = p_flat.view(num_graphs, n_ent)
            valid = ~torch.isnan(y_r)                       # identical each row
            y0 = torch.nan_to_num(y_r, nan=0.0)
            res = torch.where(valid, (p_r - y0) ** 2, torch.zeros_like(y0))
            col_sum = y0.sum(dim=0)
            a["sy"] = col_sum if a["sy"] is None else a["sy"] + col_sum
            a["sy2"] += (y0 ** 2).sum().item()
            a["sres"] += res.sum().item()
            a["graphs"] += num_graphs
            a["valid"] = valid[0]

        with torch.no_grad():
            for batch in loader:
                pred_v, pred_i = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                G = batch.num_graphs

                mask_v = batch["bus"].label_mask
                # Normalized voltage loss — the space the model now trains in.
                loss_v_norm = F.mse_loss(pred_v[mask_v], batch["bus"].y[mask_v])
                # Real-Vpu voltage: un-normalize with per-bus stats, for
                # logging and for comparison against the ~0.000042 baseline.
                v_mean_full = v_mean.repeat(G)
                v_std_full = v_std.repeat(G)
                pred_v_real_full = pred_v * v_std_full + v_mean_full
                y_v_real_full = batch["bus"].y * v_std_full + v_mean_full
                loss_v_real = F.mse_loss(pred_v_real_full[mask_v], y_v_real_full[mask_v])

                mask_i = batch[line_rel].label_mask
                # Normalized current loss — same space the model trains in,
                # used for the early-stopping objective so it matches the
                # training loss scale.
                loss_i_norm = F.mse_loss(pred_i[mask_i], batch[line_rel].y[mask_i])
                # Real-Amps current: un-normalize with per-line stats.
                i_mean_full = i_mean.repeat(G)
                i_std_full = i_std.repeat(G)
                pred_i_real_full = pred_i * i_std_full + i_mean_full
                y_i_real_full = batch[line_rel].y * i_std_full + i_mean_full
                loss_i_real = F.mse_loss(pred_i_real_full[mask_i], y_i_real_full[mask_i])

                total_v += loss_v_real.item() * G
                total_v_norm += loss_v_norm.item() * G
                total_i_real += loss_i_real.item() * G
                total_i_norm += loss_i_norm.item() * G
                n += G

                # Per-entity R^2 stats, in real units (Vpu / Amps).
                _accumulate(acc["v"], y_v_real_full, pred_v_real_full, G)
                _accumulate(acc["i"], y_i_real_full, pred_i_real_full, G)
        model.train()

        def _r2(a):
            sy_e = a["sy"][a["valid"]]
            ss_tot = a["sy2"] - (sy_e * sy_e / a["graphs"]).sum().item()
            return 1.0 - a["sres"] / ss_tot if ss_tot > 0 else float("nan")

        return {
            "mse_v": total_v / n,
            "mse_v_norm": total_v_norm / n,
            "mse_i": total_i_real / n,
            "mse_i_norm": total_i_norm / n,
            "r2_v": _r2(acc["v"]),   # per-bus reference
            "r2_i": _r2(acc["i"]),   # per-line reference
        }

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_CSV.open("w", newline="", encoding="utf-8")
    results_writer = csv.DictWriter(results_file, fieldnames=RESULTS_CSV_FIELDS)
    results_writer.writeheader()
    results_file.flush()
    print(f"writing per-epoch results to {RESULTS_CSV.resolve()}")

    def log_results(split, epoch, evaluation, train_v=None, train_i=None,
                    combined=None, is_best=None):
        """Append one row and flush. Train columns are blank for the final
        best-model rows, which report an evaluated split only."""
        def num(x):
            return "" if x is None else f"{x:.10g}"
        results_writer.writerow({
            "split": split,
            "epoch": epoch,
            "train_mse_v_vpu2": num(train_v),
            "train_mse_i_a2": num(train_i),
            "eval_mse_v_vpu2": num(evaluation["mse_v"]),
            "eval_mse_i_a2": num(evaluation["mse_i"]),
            "eval_mse_v_norm": num(evaluation["mse_v_norm"]),
            "eval_mse_i_norm": num(evaluation["mse_i_norm"]),
            "eval_r2_v": num(evaluation["r2_v"]),
            "eval_r2_i": num(evaluation["r2_i"]),
            "eval_combined": num(combined),
            "is_best": "" if is_best is None else int(bool(is_best)),
        })
        results_file.flush()

    best_combined = float("inf")
    best_state = None
    best_epoch = -1
    epochs_since_best = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss_v = 0.0
        total_loss_i = 0.0

        for batch in train_loader:
            opt.zero_grad()
            pred_v, pred_i = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)

            # Loss for Voltage — target is now per-BUS normalized by the
            # dataset (see fit_normalization/get), same treatment as current,
            # so the two loss terms sit on a comparable scale and
            # LAMBDA_CURRENT actually controls their relative weight.
            mask_v = batch["bus"].label_mask
            loss_v = F.mse_loss(pred_v[mask_v], batch["bus"].y[mask_v])

            # Loss for Current — target is already per-line normalized by
            # the dataset (see fit_normalization/get), so no separate
            # global mean_i/std_i is needed here anymore.
            mask_i = batch[line_rel].label_mask
            loss_i = F.mse_loss(pred_i[mask_i], batch[line_rel].y[mask_i])

            # Joint optimization
            loss = loss_v + (LAMBDA_CURRENT * loss_i)
            loss.backward()
            opt.step()

            # Track separately, both un-normalized back to real units
            # (Vpu / Amps) for logging against the trivial baselines.
            with torch.no_grad():
                v_mean_b = v_mean.repeat(batch.num_graphs)[mask_v]
                v_std_b = v_std.repeat(batch.num_graphs)[mask_v]
                pred_v_real = pred_v[mask_v] * v_std_b + v_mean_b
                y_v_real = batch["bus"].y[mask_v] * v_std_b + v_mean_b
                real_mse_v = F.mse_loss(pred_v_real, y_v_real)
                total_loss_v += real_mse_v.item() * batch.num_graphs

                i_mean_b = i_mean.repeat(batch.num_graphs)[mask_i]
                i_std_b = i_std.repeat(batch.num_graphs)[mask_i]
                pred_i_real = pred_i[mask_i] * i_std_b + i_mean_b
                y_i_real = batch[line_rel].y[mask_i] * i_std_b + i_mean_b
                real_mse_i = F.mse_loss(pred_i_real, y_i_real)
                total_loss_i += real_mse_i.item() * batch.num_graphs

        avg_loss_v = total_loss_v / len(train_ds)
        avg_loss_i = total_loss_i / len(train_ds)
        val = evaluate(val_loader)
        val_v, val_i = val["mse_v"], val["mse_i"]

        # Early-stopping objective: the same weighted sum the model trains
        # on. Both terms are now in normalized space, so this is a genuinely
        # balanced criterion rather than one dominated by the current head.
        combined = val["mse_v_norm"] + LAMBDA_CURRENT * val["mse_i_norm"]
        improved = combined < best_combined
        if improved:
            best_combined = combined
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        print(f"epoch {epoch}: train MSE V {avg_loss_v:.5f} | train MSE I {avg_loss_i:.5f} "
              f"| val MSE V {val_v:.5f} | val MSE I {val_i:.5f} "
              f"| val R2 V {val['r2_v']:.4f} | val R2 I {val['r2_i']:.4f} "
              f"| combined {combined:.5f}{'  <-- best' if improved else ''}")

        log_results("val", epoch, val, train_v=avg_loss_v, train_i=avg_loss_i,
                    combined=combined, is_best=improved)

        if epochs_since_best >= PATIENCE:
            print(f"early stop: no improvement in {PATIENCE} epochs "
                  f"(best epoch {best_epoch}, combined {best_combined:.5f})")
            break

    # Restore the best epoch's weights (in memory only), so the final
    # evaluation uses the checkpoint that generalized best on val — not the
    # final epoch, which (as observed) can be a worse, more overfit one.
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored best weights from epoch {best_epoch}")

    # Final evaluation on the untouched TEST split. The val split is no longer
    # an unbiased estimate — it drove early stopping — so the test split (the
    # last chronological chunk of the day, never seen in training or model
    # selection) is the honest measure of how well the model generalizes.
    val = evaluate(val_loader)
    test = evaluate(test_loader)
    print("\n=== best model (epoch {}) ===".format(best_epoch))
    print(f"val  : MSE V {val['mse_v']:.5f} | MSE I {val['mse_i']:.5f} "
          f"| R2 V {val['r2_v']:.4f} | R2 I {val['r2_i']:.4f}")
    print(f"test : MSE V {test['mse_v']:.5f} | MSE I {test['mse_i']:.5f} "
          f"| R2 V {test['r2_v']:.4f} | R2 I {test['r2_i']:.4f}")
    print("(MSE I in Amps^2; R2 is per-entity: R2 V vs each bus's own mean, "
          "R2 I vs each line's own mean; R2 <= 0 means no better than the mean)")

    # Best-epoch weights are already restored above, so these two rows describe
    # the model actually being reported, not the last epoch trained.
    log_results("val_best", best_epoch, val)
    log_results("test_best", best_epoch, test)
    results_file.close()
    print(f"results written to {RESULTS_CSV.resolve()}")
