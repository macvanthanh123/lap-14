"""
main.py — AI Evaluation Factory
Chạy benchmark V1 vs V2, tính đầy đủ metrics, sinh reports.
"""
import asyncio
import json
import os
import time
import random
from typing import Dict, List, Optional, Tuple

from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from agent.main_agent import MainAgent, MainAgentV2

# ---------------------------------------------------------------------------
# RAGAS-style Evaluator (mock — không cần gọi API)
# ---------------------------------------------------------------------------
class RAGASEvaluator:
    """
    Đánh giá Faithfulness và Relevancy bằng heuristic đơn giản.
    Trong production: thay bằng ragas.evaluate() thực.
    """

    async def score(self, test_case: Dict, response: Dict) -> Dict:
        await asyncio.sleep(0.01)  # giả lập latency tính toán

        answer = response.get("answer", "")
        expected = test_case.get("expected_answer", "")
        contexts = response.get("contexts", [])

        # Faithfulness: tỉ lệ từ trong answer xuất hiện trong context
        context_blob = " ".join(contexts).lower()
        answer_words = set(answer.lower().split())
        overlap = sum(1 for w in answer_words if w in context_blob)
        faithfulness = min(overlap / max(len(answer_words), 1), 1.0)
        faithfulness = round(0.5 + faithfulness * 0.5, 4)  # scale về [0.5, 1.0]

        # Relevancy: so sánh với expected answer
        expected_words = set(expected.lower().split())
        rel_overlap = len(answer_words & expected_words)
        relevancy = min(rel_overlap / max(len(expected_words), 1), 1.0)
        relevancy = round(0.4 + relevancy * 0.6, 4)

        return {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            # retrieval sẽ được ghi đè bởi BenchmarkRunner
            "retrieval": {"hit_rate": 0.0, "mrr": 0.0},
        }


# ---------------------------------------------------------------------------
# Multi-Judge Mock (không gọi API — dùng heuristic để demo)
# ---------------------------------------------------------------------------
class MockMultiJudge:
    """
    Mock judge dùng heuristic để không cần API key.
    Trong production: thay bằng LLMJudge từ engine/llm_judge.py.
    """

    async def _judge_single(self, question: str, answer: str,
                             ground_truth: str, model_name: str) -> Dict:
        await asyncio.sleep(random.uniform(0.02, 0.08))

        gt_words = set(ground_truth.lower().split())
        ans_words = set(answer.lower().split())

        overlap_ratio = len(gt_words & ans_words) / max(len(gt_words), 1)

        # Tính điểm có noise nhỏ để 2 judge cho điểm khác nhau
        base_score = 2.0 + overlap_ratio * 3.0
        noise = random.uniform(-0.3, 0.3) if model_name == "judge_b" else 0.0
        score = round(min(max(base_score + noise, 1.0), 5.0), 2)

        return {
            "accuracy": round(min(max(base_score * 0.9 + noise * 0.5, 1), 5), 2),
            "professionalism": round(min(max(3.5 + noise, 1), 5), 2),
            "safety": 5.0,
            "score": score,
            "reasoning": f"[{model_name}] overlap={overlap_ratio:.2f}, score={score}",
        }

    async def evaluate_multi_judge(self, question: str, answer: str,
                                    ground_truth: str) -> Dict:
        judge_a_task = self._judge_single(question, answer, ground_truth, "judge_a_gpt4o")
        judge_b_task = self._judge_single(question, answer, ground_truth, "judge_b_gpt4o_mini")

        judge_a, judge_b = await asyncio.gather(judge_a_task, judge_b_task)

        score_a = judge_a["score"]
        score_b = judge_b["score"]

        disagreement = abs(score_a - score_b)
        final_score = round((score_a + score_b) / 2, 2)
        agreement_rate = round(max(0.0, 1.0 - disagreement / 4.0), 4)

        result = {
            "final_score": final_score,
            "agreement_rate": agreement_rate,
            "individual_scores": {"judge_a": score_a, "judge_b": score_b},
            "judge_details": {"judge_a": judge_a, "judge_b": judge_b},
            "reasoning": f"Avg of judge_a={score_a} and judge_b={score_b}",
        }

        if disagreement > 1.0:
            result["needs_human_review"] = True
            result["warning"] = "Large disagreement between judges — needs human review"

        return result


# ---------------------------------------------------------------------------
# Release Gate Logic
# ---------------------------------------------------------------------------
RELEASE_THRESHOLDS = {
    "min_avg_score": 3.0,        # điểm judge tối thiểu
    "min_hit_rate": 0.60,        # hit rate retrieval tối thiểu
    "min_agreement_rate": 0.65,  # độ đồng thuận judge tối thiểu
    "max_regression_delta": -0.1, # cho phép giảm tối đa 0.1 điểm
}


def release_gate(v1_metrics: Dict, v2_metrics: Dict) -> Tuple[str, str]:
    """
    Tự động quyết định Release hoặc Rollback.
    Trả về (decision, reason).
    """
    reasons = []
    block = False

    delta_score = v2_metrics["avg_score"] - v1_metrics["avg_score"]
    delta_hit = v2_metrics["hit_rate"] - v1_metrics["hit_rate"]

    if v2_metrics["avg_score"] < RELEASE_THRESHOLDS["min_avg_score"]:
        block = True
        reasons.append(f"avg_score={v2_metrics['avg_score']:.2f} < threshold {RELEASE_THRESHOLDS['min_avg_score']}")

    if v2_metrics["hit_rate"] < RELEASE_THRESHOLDS["min_hit_rate"]:
        block = True
        reasons.append(f"hit_rate={v2_metrics['hit_rate']:.2f} < threshold {RELEASE_THRESHOLDS['min_hit_rate']}")

    if delta_score < RELEASE_THRESHOLDS["max_regression_delta"]:
        block = True
        reasons.append(f"score regression delta={delta_score:.3f} vượt ngưỡng cho phép")

    if block:
        return "ROLLBACK", " | ".join(reasons)

    positives = []
    if delta_score > 0:
        positives.append(f"score +{delta_score:.3f}")
    if delta_hit > 0:
        positives.append(f"hit_rate +{delta_hit:.3f}")

    return "RELEASE", "Tất cả chỉ số đạt ngưỡng. " + (", ".join(positives) if positives else "Không regression.")


# ---------------------------------------------------------------------------
# Benchmark runner helper
# ---------------------------------------------------------------------------
async def run_benchmark(
    agent, dataset: List[Dict], version_label: str
) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
    """Chạy đầy đủ benchmark, trả về (results, summary)."""
    evaluator = RAGASEvaluator()
    judge = MockMultiJudge()
    retrieval_eval = RetrievalEvaluator()

    runner = BenchmarkRunner(agent, evaluator, judge, retrieval_eval)
    start = time.perf_counter()
    results = await runner.run_all(dataset)
    elapsed = time.perf_counter() - start

    total = len(results)
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = total - passed

    avg_score = sum(r["judge"]["final_score"] for r in results) / total
    avg_hit_rate = sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total
    avg_mrr = sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total
    avg_agreement = sum(r["judge"]["agreement_rate"] for r in results) / total
    avg_faithfulness = sum(r["ragas"]["faithfulness"] for r in results) / total
    avg_relevancy = sum(r["ragas"]["relevancy"] for r in results) / total
    avg_latency = sum(r["latency"] for r in results) / total
    total_tokens = sum(r.get("tokens_used", 0) for r in results)

    # Tính Agreement Rate tổng thể (Cohen's Kappa approx)
    scores_a = [r["judge"]["individual_scores"]["judge_a"] for r in results]
    scores_b = [r["judge"]["individual_scores"]["judge_b"] for r in results]
    mean_a = sum(scores_a) / total
    mean_b = sum(scores_b) / total
    cohen_kappa_approx = round(1 - abs(mean_a - mean_b) / 4, 4)

    summary = {
        "metadata": {
            "version": version_label,
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total, 4),
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": round(avg_score, 4),
            "hit_rate": round(avg_hit_rate, 4),
            "avg_mrr": round(avg_mrr, 4),
            "agreement_rate": round(avg_agreement, 4),
            "cohen_kappa_approx": cohen_kappa_approx,
            "avg_faithfulness": round(avg_faithfulness, 4),
            "avg_relevancy": round(avg_relevancy, 4),
            "avg_latency_sec": round(avg_latency, 4),
            "total_tokens_used": total_tokens,
            "estimated_cost_usd": round(total_tokens * 0.000002, 6),  # ~$0.002/1K tokens
        },
    }

    return results, summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main():
    print("=" * 60)
    print("🏭 AI EVALUATION FACTORY — Lab Day 14")
    print("=" * 60)

    # --- Load dataset ---
    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng.")
        return

    print(f"📂 Loaded {len(dataset)} test cases từ golden_set.jsonl\n")

    # --- Benchmark V1 ---
    print("🔵 Chạy Benchmark Agent V1 (Base)...")
    agent_v1 = MainAgent(version="v1", top_k=3)
    v1_results, v1_summary = await run_benchmark(agent_v1, dataset, "Agent_V1_Base")
    if not v1_results:
        print("❌ V1 benchmark thất bại.")
        return
    print(f"   V1 avg_score={v1_summary['metrics']['avg_score']:.3f}  "
          f"hit_rate={v1_summary['metrics']['hit_rate']:.3f}  "
          f"elapsed={v1_summary['metadata']['elapsed_seconds']}s\n")

    # --- Benchmark V2 ---
    print("🟢 Chạy Benchmark Agent V2 (Optimized)...")
    agent_v2 = MainAgentV2()
    v2_results, v2_summary = await run_benchmark(agent_v2, dataset, "Agent_V2_Optimized")
    if not v2_results:
        print("❌ V2 benchmark thất bại.")
        return
    print(f"   V2 avg_score={v2_summary['metrics']['avg_score']:.3f}  "
          f"hit_rate={v2_summary['metrics']['hit_rate']:.3f}  "
          f"elapsed={v2_summary['metadata']['elapsed_seconds']}s\n")

    # --- Regression Analysis ---
    print("📊 " + "─" * 50)
    print("   REGRESSION ANALYSIS: V1 vs V2")
    print("─" * 54)
    m1, m2 = v1_summary["metrics"], v2_summary["metrics"]
    rows = [
        ("avg_score",        m1["avg_score"],        m2["avg_score"]),
        ("hit_rate",         m1["hit_rate"],          m2["hit_rate"]),
        ("avg_mrr",          m1["avg_mrr"],           m2["avg_mrr"]),
        ("agreement_rate",   m1["agreement_rate"],    m2["agreement_rate"]),
        ("avg_faithfulness", m1["avg_faithfulness"],  m2["avg_faithfulness"]),
        ("avg_relevancy",    m1["avg_relevancy"],      m2["avg_relevancy"]),
        ("avg_latency_sec",  m1["avg_latency_sec"],   m2["avg_latency_sec"]),
    ]
    print(f"{'Metric':<22} {'V1':>8} {'V2':>8} {'Delta':>8}")
    print("─" * 54)
    for name, v1_val, v2_val in rows:
        delta = v2_val - v1_val
        sign = "+" if delta >= 0 else ""
        print(f"{name:<22} {v1_val:>8.4f} {v2_val:>8.4f} {sign}{delta:>7.4f}")
    print("─" * 54)

    # --- Release Gate ---
    decision, reason = release_gate(m1, m2)
    decision_icon = "✅" if decision == "RELEASE" else "❌"
    print(f"\n{decision_icon} RELEASE GATE DECISION: {decision}")
    print(f"   Reason: {reason}")

    # Thêm regression & decision vào summary
    delta_score = m2["avg_score"] - m1["avg_score"]
    v2_summary["regression"] = {
        "v1_avg_score": m1["avg_score"],
        "v2_avg_score": m2["avg_score"],
        "delta_score": round(delta_score, 4),
        "delta_hit_rate": round(m2["hit_rate"] - m1["hit_rate"], 4),
        "decision": decision,
        "reason": reason,
    }
    v2_summary["cost_analysis"] = {
        "v1_tokens": m1["total_tokens_used"],
        "v2_tokens": m2["total_tokens_used"],
        "v1_cost_usd": m1["estimated_cost_usd"],
        "v2_cost_usd": m2["estimated_cost_usd"],
        "cost_reduction_pct": round(
            (1 - m2["total_tokens_used"] / max(m1["total_tokens_used"], 1)) * 100, 2
        ),
    }

    # --- Lưu reports ---
    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    # Lưu thêm full comparison
    comparison = {"v1": v1_summary, "v2": v2_summary}
    with open("reports/regression_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Reports saved:")
    print("   reports/summary.json")
    print("   reports/benchmark_results.json")
    print("   reports/regression_comparison.json")
    print("\n✅ Benchmark hoàn thành!")


if __name__ == "__main__":
    asyncio.run(main())
