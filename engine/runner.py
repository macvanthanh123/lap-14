import asyncio
import time
from typing import List, Dict, Optional
from engine.retrieval_eval import RetrievalEvaluator


class BenchmarkRunner:
    """
    Async benchmark runner với:
    - Batch processing để tránh rate limit
    - Tích hợp RetrievalEvaluator (Hit Rate + MRR)
    - Tích hợp Multi-Judge consensus
    - Báo cáo cost & token usage
    """

    def __init__(self, agent, evaluator, judge, retrieval_evaluator: Optional[RetrievalEvaluator] = None):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.retrieval_evaluator = retrieval_evaluator or RetrievalEvaluator()

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()

        # 1. Gọi Agent — nhận về answer + retrieved_ids
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time

        # 2. Tính Retrieval Metrics (Hit Rate & MRR) cho case này
        expected_ids = test_case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("retrieved_ids", [])

        hit_rate = self.retrieval_evaluator.calculate_hit_rate(
            expected_ids, retrieved_ids, top_k=3
        )
        mrr = self.retrieval_evaluator.calculate_mrr(expected_ids, retrieved_ids)

        # 3. Chạy RAGAS-style metrics (faithfulness, relevancy)
        ragas_scores = await self.evaluator.score(test_case, response)
        # Ghi đè retrieval metrics bằng giá trị thực tính được
        ragas_scores["retrieval"] = {"hit_rate": hit_rate, "mrr": mrr}

        # 4. Chạy Multi-Judge
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case.get("expected_answer", ""),
        )

        tokens_used = response.get("metadata", {}).get("tokens_used", 0)

        return {
            "test_case": test_case["question"],
            "expected_answer": test_case.get("expected_answer", ""),
            "agent_response": response["answer"],
            "retrieved_ids": retrieved_ids,
            "expected_retrieval_ids": expected_ids,
            "latency": round(latency, 4),
            "tokens_used": tokens_used,
            "ragas": ragas_scores,
            "judge": judge_result,
            "metadata": test_case.get("metadata", {}),
            "status": "fail" if judge_result["final_score"] < 3 else "pass",
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 10) -> List[Dict]:
        """
        Chạy song song theo batch để không bị rate limit.
        batch_size=10 tối ưu cho mock environment.
        """
        results = []
        total = len(dataset)
        for i in range(0, total, batch_size):
            batch = dataset[i:i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            print(f"  ⚡ Processed {min(i + batch_size, total)}/{total} cases...")
        return results
