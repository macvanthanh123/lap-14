from typing import List, Dict


class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: int = 3,
    ) -> float:
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
    ) -> float:

        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)

        return 0.0

    async def evaluate_batch(
        self,
        dataset: List[Dict],
        top_k: int = 3,
    ) -> Dict:

        if not dataset:
            return {
                "avg_hit_rate": 0.0,
                "avg_mrr": 0.0,
                "num_samples": 0,
            }

        hit_rates = []
        mrr_scores = []

        for item in dataset:
            expected_ids = item.get(
                "expected_retrieval_ids",
                [],
            )

            retrieved_ids = item.get(
                "retrieved_ids",
                [],
            )

            hit_rate = self.calculate_hit_rate(
                expected_ids,
                retrieved_ids,
                top_k,
            )

            mrr = self.calculate_mrr(
                expected_ids,
                retrieved_ids,
            )

            hit_rates.append(hit_rate)
            mrr_scores.append(mrr)

        avg_hit_rate = sum(hit_rates) / len(hit_rates)
        avg_mrr = sum(mrr_scores) / len(mrr_scores)

        return {
            "avg_hit_rate": round(avg_hit_rate, 4),
            "avg_mrr": round(avg_mrr, 4),
            "num_samples": len(dataset),
            "details": {
                "hit_rates": hit_rates,
                "mrr_scores": mrr_scores,
            },
        }