"""
RAG Evaluation Tool
====================
Measure faithfulness, relevancy, precision, and answer quality.

Usage:
    from src.tools.evaluator import RAGEvaluator
    evaluator = RAGEvaluator()
    scores = evaluator.evaluate(question, answer, contexts, ground_truth)
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass

from src.utils.logger import log


@dataclass
class EvalResult:
    """Evaluation metrics for a single query."""
    faithfulness: float = 0.0
    relevancy: float = 0.0
    context_precision: float = 0.0
    answer_correctness: float = 0.0
    overall: float = 0.0


class RAGEvaluator:
    """Evaluate RAG pipeline quality using multiple metrics."""

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvalResult:
        """Run all evaluation metrics."""
        result = EvalResult()

        result.faithfulness = self._score_faithfulness(answer, contexts)
        result.relevancy = self._score_relevancy(question, answer)
        result.context_precision = self._score_context_precision(question, contexts)
        
        if ground_truth:
            result.answer_correctness = self._score_correctness(answer, ground_truth)

        scores = [result.faithfulness, result.relevancy, result.context_precision]
        if ground_truth:
            scores.append(result.answer_correctness)
        result.overall = sum(scores) / len(scores)

        return result

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate a batch of queries and return aggregate scores."""
        results = []
        for i in range(len(questions)):
            gt = ground_truths[i] if ground_truths else None
            r = self.evaluate(questions[i], answers[i], contexts_list[i], gt)
            results.append(r)

        avg = lambda attr: sum(getattr(r, attr) for r in results) / len(results)
        return {
            "faithfulness": round(avg("faithfulness"), 4),
            "relevancy": round(avg("relevancy"), 4),
            "context_precision": round(avg("context_precision"), 4),
            "answer_correctness": round(avg("answer_correctness"), 4),
            "overall": round(avg("overall"), 4),
            "num_queries": len(results),
        }

    @staticmethod
    def _score_faithfulness(answer: str, contexts: List[str]) -> float:
        """Score how well the answer is grounded in the retrieved contexts."""
        if not contexts or not answer:
            return 0.0
        
        answer_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        context_text = " ".join(contexts).lower()
        context_words = set(re.findall(r'\b\w{4,}\b', context_text))
        
        if not answer_words:
            return 0.0
        
        overlap = answer_words & context_words
        return len(overlap) / len(answer_words)

    @staticmethod
    def _score_relevancy(question: str, answer: str) -> float:
        """Score how relevant the answer is to the question."""
        if not answer:
            return 0.0
        
        q_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
        a_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        
        if not q_words:
            return 0.5
        
        overlap = q_words & a_words
        return min(1.0, len(overlap) / len(q_words) + 0.3)

    @staticmethod
    def _score_context_precision(question: str, contexts: List[str]) -> float:
        """Score how precise the retrieved contexts are for the question."""
        if not contexts:
            return 0.0
        
        q_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
        relevant = 0
        
        for ctx in contexts:
            ctx_words = set(re.findall(r'\b\w{4,}\b', ctx.lower()))
            if q_words & ctx_words:
                relevant += 1
        
        return relevant / len(contexts)

    @staticmethod
    def _score_correctness(answer: str, ground_truth: str) -> float:
        """Score answer correctness against ground truth."""
        gt_words = set(re.findall(r'\b\w{4,}\b', ground_truth.lower()))
        ans_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        
        if not gt_words:
            return 0.5
        
        overlap = gt_words & ans_words
        precision = len(overlap) / len(ans_words) if ans_words else 0
        recall = len(overlap) / len(gt_words)
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)  # F1
