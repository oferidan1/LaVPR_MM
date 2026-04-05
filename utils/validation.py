import numpy as np
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable
import torch

def get_validation_recalls(r_list, q_list, k_values, gt, print_results=True, faiss_gpu=False, dataset_name='dataset without name ?'):
        
        embed_size = r_list.shape[1]
        if faiss_gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
        # build index
        else:
            faiss_index = faiss.IndexFlatL2(embed_size)
        
        # add references
        r_list = r_list.to(torch.float32)
        q_list = q_list.to(torch.float32)

        faiss_index.add(r_list)

        # search for queries in the index
        _, predictions = faiss_index.search(q_list, max(k_values))
        
        
        
        # start calculating recall_at_k
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.isin(pred[:n], gt[q_idx])):
                    correct_at_k[i:] += 1
                    break
        
        correct_at_k = correct_at_k / len(predictions)
        d = {k:v for (k,v) in zip(k_values, correct_at_k)}

        if print_results:
            print() # print a new line
            table = PrettyTable()
            table.field_names = ['K']+[str(k) for k in k_values]
            table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in correct_at_k])
            print(table.get_string(title=f"Performances on {dataset_name}"))
        
        return d


def get_validation_recalls_dynamic_fusion(r_list, q_list, r_text_list, q_text_list, w_r, w_q, k_values, gt, rrf_k=60.0, print_results=True, faiss_gpu=False, dataset_name='dataset without name ?'):

        embed_size = r_list.shape[1]
        embed_text_size = r_text_list.shape[1]
        if faiss_gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index_i = faiss.GpuIndexFlatIP(res, embed_size, flat_config)
        else:
            faiss_index_i = faiss.IndexFlatIP(embed_size)
            faiss_index_t = faiss.IndexFlatIP(embed_text_size)

        r_list = r_list.to(torch.float32)
        q_list = q_list.to(torch.float32)
        r_text_list = r_text_list.to(torch.float32)
        q_text_list = q_text_list.to(torch.float32)

        faiss_index_i.add(r_list)
        faiss_index_t.add(r_text_list)

        max_k = min(10000, len(r_list))
        _, predictions_v = faiss_index_i.search(q_list, max_k)
        _, predictions_t = faiss_index_t.search(q_text_list, max_k)

        predictions = rerank_predictions_rrf(
            predictions_v, predictions_t, w_r, w_q,
            rrf_k=rrf_k, max_results=max(k_values),
        )

        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                if np.any(np.isin(pred[:n], gt[q_idx])):
                    correct_at_k[i:] += 1
                    break

        correct_at_k = correct_at_k / len(predictions)
        d = {k:v for (k,v) in zip(k_values, correct_at_k)}

        if print_results:
            print()
            table = PrettyTable()
            table.field_names = ['K']+[str(k) for k in k_values]
            table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in correct_at_k])
            print(table.get_string(title=f"W-RRF (k={rrf_k}) on {dataset_name}"))

        return d


def rerank_predictions_rrf(vision_predictions, text_predictions, w_r, w_q,
                           rrf_k, max_results):
    """Weighted Reciprocal Rank Fusion (SW-RRF).

    SW-RRF(d) = w_v_avg / (k + r_v(d))  +  w_l_avg / (k + r_l(d))

    Ranks are 1-indexed (best candidate = rank 1).
    Pairwise weights are averaged: w_v_avg = (w_v_query + w_v_ref) / 2.
    """
    if isinstance(w_r, torch.Tensor):
        w_r = w_r.numpy()
    if isinstance(w_q, torch.Tensor):
        w_q = w_q.numpy()
    if isinstance(vision_predictions, torch.Tensor):
        vision_predictions = vision_predictions.numpy()
    if isinstance(text_predictions, torch.Tensor):
        text_predictions = text_predictions.numpy()

    num_queries = vision_predictions.shape[0]
    max_k = vision_predictions.shape[1]
    fallback_rank = max_k + 1

    # Normalise fixed-weighting (scalar alpha) to [N, 2]
    if w_r.ndim == 1:
        alpha = float(w_r.mean())
        num_refs = int(vision_predictions.max()) + 1
        w_r = np.column_stack([np.full(num_refs, alpha),
                               np.full(num_refs, 1.0 - alpha)])
    if w_q.ndim == 1:
        alpha = float(w_q.mean())
        w_q = np.column_stack([np.full(num_queries, alpha),
                               np.full(num_queries, 1.0 - alpha)])

    print(f"W-RRF | k={rrf_k} | query alpha(w_v): "
          f"mean={w_q[:, 0].mean():.4f}  std={w_q[:, 0].std():.4f}")

    all_predictions = np.zeros((num_queries, max_results), dtype=np.int64)

    for q_idx in range(num_queries):
        v_rank = {}
        for rank_0, ref_id in enumerate(vision_predictions[q_idx]):
            ref_id = int(ref_id)
            if ref_id >= 0:
                v_rank[ref_id] = rank_0 + 1

        t_rank = {}
        for rank_0, ref_id in enumerate(text_predictions[q_idx]):
            ref_id = int(ref_id)
            if ref_id >= 0:
                t_rank[ref_id] = rank_0 + 1

        all_refs = set(v_rank.keys()) | set(t_rank.keys())

        w_v_q = w_q[q_idx, 0]
        w_l_q = w_q[q_idx, 1]

        rrf_scores = {}
        for ref in all_refs:
            rv = v_rank.get(ref, fallback_rank)
            rl = t_rank.get(ref, fallback_rank)
            w_v_avg = (w_r[ref, 0] + w_v_q) / 2.0
            w_l_avg = (w_r[ref, 1] + w_l_q) / 2.0
            rrf_scores[ref] = w_v_avg / (rrf_k + rv) + w_l_avg / (rrf_k + rl)

        sorted_refs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        for i in range(min(max_results, len(sorted_refs))):
            all_predictions[q_idx, i] = sorted_refs[i][0]

    return all_predictions

