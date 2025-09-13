import itertools
from typing import List, Dict, Optional, Type

from torch import nn
from statistics import mean
from sentence_transformers import util
from torch.utils.tensorboard import SummaryWriter


class Evaluator:
    def __init__(self, metrics_at_k: Dict[str, List[int]] = {'recall': [100, 200, 500], 'map': [100], 'mrr': [100]}):
        self.metrics_at_k = metrics_at_k

    def compute_all_metrics(self, all_results: List[List[int]], all_ground_truths: List[List[int]]):
        scores = dict()
        for k in self.metrics_at_k['recall']:
            recall_scalar = self.compute_mean_score(self.recall, all_ground_truths, all_results, k)
            scores[f'recall@{k}'] = recall_scalar

        for k in self.metrics_at_k['map']:
            map_scalar = self.compute_mean_score(self.average_precision, all_ground_truths, all_results, k)
            scores[f'map@{k}'] = map_scalar

        for k in self.metrics_at_k['mrr']:
            mrr_scalar = self.compute_mean_score(self.reciprocal_rank, all_ground_truths, all_results, k)
            scores[f'mrr@{k}'] = mrr_scalar
        return scores

    def compute_mean_score(self, func, all_ground_truths: List[List[int]], all_results: List[List[int]], k: int = None):
        return mean([func(truths, res, k) for truths, res in zip(all_ground_truths, all_results)])

    def precision(self, ground_truths: List[int], results: List[int], k: int = None):
        k = len(results) if k is None else k
        relevances = [1 if d in ground_truths else 0 for d in results[:k]]
        return sum(relevances)/len(results[:k])

    def recall(self, ground_truths: List[int], results: List[int], k: int = None):
        k = len(results) if k is None else k
        relevances = [1 if d in ground_truths else 0 for d in results[:k]]
        return sum(relevances)/len(ground_truths)

    def fscore(self, ground_truths: List[int], results: List[int], k: int = None):
        p = self.precision(ground_truths, results, k)
        r = self.recall(ground_truths, results, k)
        return (2*p*r)/(p+r) if (p != 0.0 or r != 0.0) else 0.0

    def reciprocal_rank(self, ground_truths: List[int], results: List[int], k: int = None):
        k = len(results) if k is None else k
        return max([1/(i+1) if d in ground_truths else 0.0 for i, d in enumerate(results[:k])])

    def average_precision(self, ground_truths: List[int], results: List[int], k: int = None):
        k = len(results) if k is None else k
        p_k = [self.precision(ground_truths, results, k=i+1) if d in ground_truths else 0 for i, d in enumerate(results[:k])]
        return sum(p_k)/len(ground_truths)



class BiEncoderEvaluator(Evaluator):
    def __init__(self, 
                 queries: Dict[str, str], #qid -> query
                 documents: Dict[str, str],  #doc_id -> doc
                 relevant_pairs: Dict[str, List[int]], # qid -> List[doc_id]
                 score_fn: str,
                 metrics_at_k: Dict[str, List[int]] = {'recall': [100, 200, 500], 'map': [100], 'mrr': [100]},
        ):
        super().__init__(metrics_at_k)
        assert score_fn in ['dot', 'cos'], f"Unknown score function: {score_fn}"
        self.score_fn = util.dot_score if score_fn == 'dot' else util.cos_sim
        self.query_ids = list(queries.keys())
        self.queries = [queries[qid] for qid in self.query_ids]
        self.document_ids = list(documents.keys())
        self.documents = [documents[doc_id] for doc_id in self.document_ids]
        self.relevant_pairs = relevant_pairs
    

    def __call__(self, 
                 model: Type[nn.Module], 
                 device: str, 
                 batch_size: int, 
                 writer: Optional[Type[SummaryWriter]] = None, 
                 epoch: Optional[int] = None
        ):
        # Encode queries.
        q_embeddings = model.q_encoder.encode(texts=self.queries, device=device, batch_size=batch_size)
        d_embeddings = model.d_encoder.encode(texts=self.documents, device=device, batch_size=batch_size)

        # Retrieve top candidates -> returns a List[List[Dict[str,int]]].
        all_results = util.semantic_search(
            query_embeddings=q_embeddings, 
            corpus_embeddings=d_embeddings,
            top_k=max(list(itertools.chain(*self.metrics_at_k.values()))),
            score_function=self.score_fn)
        all_results = [[self.document_ids[result['corpus_id']] for result in results] for results in all_results] #Extract the doc_id only -> List[List[int]] (NB: +1 because article ids start at 1 while semantic_search returns indices in the given list).

        # Get ground truths.
        all_ground_truths = [self.relevant_pairs[qid] for qid in self.query_ids]

        # Compute metrics.
        scores = dict()
        for k in self.metrics_at_k['recall']:
            recall_scalar = self.compute_mean_score(self.recall, all_ground_truths, all_results, k)
            if writer is not None:
                writer.add_scalar(f'Val/recall/recall_at_{k}', recall_scalar, epoch)
            scores[f'recall@{k}'] = recall_scalar

        for k in self.metrics_at_k['map']:
            map_scalar = self.compute_mean_score(self.average_precision, all_ground_truths, all_results, k)
            if writer is not None:
                writer.add_scalar(f'Val/map/map_at_{k}', map_scalar, epoch)
            scores[f'map@{k}'] = map_scalar

        for k in self.metrics_at_k['mrr']:
            mrr_scalar = self.compute_mean_score(self.reciprocal_rank, all_ground_truths, all_results, k)
            if writer is not None:
                writer.add_scalar(f'Val/mrr/mrr_at_{k}', mrr_scalar, epoch)
            scores[f'mrr@{k}'] = mrr_scalar

        k = 3
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for retr, gnd in zip(all_results, all_ground_truths):
            retrieved = set(retr[:k])
            ground_truth = set(gnd)

            tp = len(retrieved & ground_truth)
            fp = len(retrieved - ground_truth)
            fn = len(ground_truth - retrieved)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1_scores.append(f1)

        # Macro-averaged F1 (average F1 scores across queries)
        macro_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0
        
        # Micro-averaged F1 (aggregate TP, FP, FN across queries)
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0
        scores["macro_presion"] = sum(all_precisions) / len(all_precisions) if all_precisions else 0
        scores["macro_recall"] = sum(all_recalls) / len(all_recalls) if all_recalls else 0
        scores["macro_f1"] = macro_f1
        scores["micro_presion"] = micro_precision
        scores["micro_recall"] = micro_recall
        scores["micro_f1"] = micro_f1

        if writer is not None:
            writer.add_scalar(f'Val/f1/macro_at_{k}', macro_f1, epoch)
            writer.add_scalar(f'Val/f1/micro_at_{k}', micro_f1, epoch)
            writer.add_scalar(f'Val/precision/macro_at_{k}', scores["macro_presion"], epoch)
            writer.add_scalar(f'Val/precision/micro_at_{k}', scores["micro_presion"], epoch)
            writer.add_scalar(f'Val/recall/macro_at_{k}', scores["macro_recall"], epoch)
            writer.add_scalar(f'Val/recall/micro_at_{k}', scores["micro_recall"], epoch)
        return scores

class BiEncoderTester():
    def __init__(self, 
                 queries: Dict[str, str], #qid -> query
                 documents: Dict[str, str],  #doc_id -> doc
                 score_fn: str,
        ):
        assert score_fn in ['dot', 'cos'], f"Unknown score function: {score_fn}"
        self.score_fn = util.dot_score if score_fn == 'dot' else util.cos_sim
        self.query_ids = list(queries.keys())
        self.queries = [queries[qid] for qid in self.query_ids]
        self.document_ids = list(documents.keys())
        self.documents = [documents[doc_id] for doc_id in self.document_ids]
    

    def __call__(self, 
                 model: Type[nn.Module], 
                 device: str, 
                 batch_size: int, 
                 writer: Optional[Type[SummaryWriter]] = None, 
                 epoch: Optional[int] = None
        ):
        # Encode queries.
        q_embeddings = model.q_encoder.encode(texts=self.queries, device=device, batch_size=batch_size)
        d_embeddings = model.d_encoder.encode(texts=self.documents, device=device, batch_size=batch_size)

        # Retrieve top candidates -> returns a List[List[Dict[str,int]]].
        all_results = util.semantic_search(
            query_embeddings=q_embeddings, 
            corpus_embeddings=d_embeddings,
            top_k=3,
            score_function=self.score_fn)
        
        all_results = [[self.document_ids[result['corpus_id']] for result in results] for results in all_results] #Extract the doc_id only -> List[List[int]] (NB: +1 because article ids start at 1 while semantic_search returns indices in the given list).
        
        predictions = []
        for id, result in zip(self.query_ids, all_results):
            predictions.append([id, ",".join(result)])
        return predictions