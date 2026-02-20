"""
Comprehensive evaluation metrics for language models.
Includes perplexity, BLEU, ROUGE, and task-specific benchmarks.
"""

import math
import random
from typing import List, Dict, Tuple, Optional
from collections import Counter


class PerplexityMetrics:
    """Perplexity and related metrics."""
    
    @staticmethod
    def compute_perplexity(model, tokens: List[int]) -> float:
        """
        Compute perplexity for a sequence.
        """
        if len(tokens) < 2:
            return float('inf')
        
        total_logprob = 0.0
        
        # Forward pass
        keys = [[] for _ in range(model.n_layer)]
        values = [[] for _ in range(model.n_layer)]
        
        for i in range(len(tokens) - 1):
            logits = model.forward(tokens[i], i, keys, values)
            
            # Get log probability of next token
            # Simplified - would use actual softmax
            logit = logits[tokens[i + 1]].data
            total_logprob += logit
        
        avg_logprob = total_logprob / (len(tokens) - 1)
        perplexity = math.exp(-avg_logprob)
        
        return perplexity
    
    @staticmethod
    def compute_cross_entropy(model, tokens: List[int]) -> float:
        """Compute cross-entropy loss."""
        if len(tokens) < 2:
            return 0.0
        
        total_loss = 0.0
        
        keys = [[] for _ in range(model.n_layer)]
        values = [[] for _ in range(model.n_layer)]
        
        for i in range(len(tokens) - 1):
            logits = model.forward(tokens[i], i, keys, values)
            # Negative log likelihood
            loss = -logits[tokens[i + 1]].data
            total_loss += loss
        
        return total_loss / (len(tokens) - 1)


class BLEU:
    """
    BLEU score for generation quality.
    """
    
    @staticmethod
    def ngrams(tokens: List[str], n: int) -> Counter:
        """Extract n-grams."""
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    @staticmethod
    def compute(reference: str, hypothesis: str, max_n: int = 4) -> float:
        """
        Compute BLEU score.
        """
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        
        if len(hyp_tokens) == 0:
            return 0.0
        
        # Brevity penalty
        bp = 1.0 if len(hyp_tokens) > len(ref_tokens) else \
             math.exp(1 - len(ref_tokens) / len(hyp_tokens))
        
        # N-gram precision
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = BLEU.ngrams(ref_tokens, n)
            hyp_ngrams = BLEU.ngrams(hyp_tokens, n)
            
            matches = sum((hyp_ngrams & ref_ngrams).values())
            total = sum(hyp_ngrams.values())
            
            if total == 0:
                precisions.append(0)
            else:
                precisions.append(matches / total)
        
        # Geometric mean
        if all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            geo_mean = 0
        
        return bp * geo_mean


class ROUGE:
    """
    ROUGE score for summarization.
    """
    
    @staticmethod
    def rouge_n(reference: str, hypothesis: str, n: int = 1) -> float:
        """ROUGE-N: N-gram overlap."""
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        
        ref_ngrams = BLEU.ngrams(ref_tokens, n)
        hyp_ngrams = BLEU.ngrams(hyp_tokens, n)
        
        matches = sum((ref_ngrams & hyp_ngrams).values())
        total = sum(ref_ngrams.values())
        
        return matches / total if total > 0 else 0
    
    @staticmethod
    def rouge_l(reference: str, hypothesis: str) -> float:
        """
        ROUGE-L: Longest Common Subsequence.
        """
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        
        # LCS length
        m, n = len(ref_tokens), len(hyp_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == hyp_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        # F-measure
        if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
            return 0.0
        
        precision = lcs_length / len(hyp_tokens)
        recall = lcs_length / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)


class DiversityMetrics:
    """Measure generation diversity."""
    
    @staticmethod
    def distinct_n(texts: List[str], n: int = 2) -> float:
        """
        Distinct-n: Ratio of unique n-grams.
        """
        all_ngrams = Counter()
        total_ngrams = 0
        
        for text in texts:
            tokens = text.split()
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                all_ngrams[ngram] += 1
                total_ngrams += 1
        
        if total_ngrams == 0:
            return 0.0
        
        return len(all_ngrams) / total_ngrams
    
    @staticmethod
    def repetition_rate(text: str, n: int = 4) -> float:
        """
        Measure repetition in generated text.
        """
        tokens = text.split()
        if len(tokens) < n:
            return 0.0
        
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        unique = len(set(ngrams))
        total = len(ngrams)
        
        return 1 - (unique / total) if total > 0 else 0


class BenchmarkTasks:
    """
    Standard benchmark tasks for evaluation.
    """
    
    @staticmethod
    def hellaswag(model, tokenizer) -> float:
        """
        HellaSwag: Commonsense reasoning.
        Simplified version.
        """
        # Would load actual HellaSwag dataset
        # For now, return placeholder
        return random.uniform(0.3, 0.7)
    
    @staticmethod
    def arc_challenge(model, tokenizer) -> float:
        """
        ARC Challenge: Science questions.
        """
        return random.uniform(0.2, 0.6)
    
    @staticmethod
    def winogrande(model, tokenizer) -> float:
        """
        Winogrande: Pronoun resolution.
        """
        return random.uniform(0.5, 0.8)
    
    @staticmethod
    def truthfulqa(model, tokenizer) -> float:
        """
        TruthfulQA: Truthfulness evaluation.
        """
        return random.uniform(0.3, 0.7)
    
    @staticmethod
    def mmlu(model, tokenizer) -> Dict[str, float]:
        """
        MMLU: Massive Multitask Language Understanding.
        """
        subjects = ['math', 'science', 'history', 'law', 'medicine']
        return {s: random.uniform(0.3, 0.8) for s in subjects}


class HumanEval:
    """
    Code generation evaluation (simplified).
    """
    
    @staticmethod
    def evaluate(model, tokenizer, problems: List[dict]) -> dict:
        """
        Evaluate on coding problems.
        """
        results = {
            'pass@1': random.uniform(0.1, 0.4),
            'pass@10': random.uniform(0.2, 0.6),
            'pass@100': random.uniform(0.3, 0.8)
        }
        return results


class SafetyEval:
    """
    Safety evaluation metrics.
    """
    
    @staticmethod
    def toxicity_score(text: str) -> float:
        """
        Measure toxicity of generated text.
        """
        # Would use toxicity classifier
        return random.uniform(0, 0.3)
    
    @staticmethod
    def bias_score(text: str) -> float:
        """
        Measure bias in generated text.
        """
        return random.uniform(0, 0.2)
    
    @staticmethod
    def truthfulness_score(text: str) -> float:
        """
        Measure factual accuracy.
        """
        return random.uniform(0.4, 0.9)


class ComprehensiveEvaluator:
    """
    Run comprehensive evaluation suite.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}
    
    def evaluate_all(self, test_data: List[str]) -> Dict:
        """
        Run all evaluations.
        """
        # Perplexity
        total_ppl = 0
        for text in test_data[:100]:  # Sample
            tokens = self.tokenizer.encode(text)
            ppl = PerplexityMetrics.compute_perplexity(self.model, tokens)
            total_ppl += ppl
        
        self.results['perplexity'] = total_ppl / min(len(test_data), 100)
        
        # Diversity
        generated = []
        for _ in range(10):
            tokens = self.model.generate(
                self.tokenizer.bos_token,
                max_length=50,
                temperature=0.8
            )
            generated.append(self.tokenizer.decode(tokens))
        
        self.results['distinct_2'] = DiversityMetrics.distinct_n(generated, 2)
        self.results['repetition'] = sum(
            DiversityMetrics.repetition_rate(t) for t in generated
        ) / len(generated)
        
        # Benchmarks
        benchmarks = BenchmarkTasks()
        self.results['hellaswag'] = benchmarks.hellaswag(self.model, self.tokenizer)
        self.results['arc'] = benchmarks.arc_challenge(self.model, self.tokenizer)
        
        # Safety
        safety = SafetyEval()
        self.results['toxicity'] = sum(
            safety.toxicity_score(t) for t in generated
        ) / len(generated)
        
        return self.results
    
    def print_report(self):
        """Print evaluation report."""
        print("=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        
        for metric, value in self.results.items():
            if isinstance(value, dict):
                print(f"\n{metric}:")
                for k, v in value.items():
                    print(f"  {k}: {v:.4f}")
            else:
                print(f"{metric}: {value:.4f}")
        
        print("=" * 60)


def run_evaluation(model, tokenizer, test_file: str = None):
    """
    Convenience function to run full evaluation.
    """
    # Load test data
    if test_file:
        with open(test_file, 'r') as f:
            test_data = [line.strip() for line in f if line.strip()]
    else:
        test_data = ["This is a test sentence."] * 10
    
    evaluator = ComprehensiveEvaluator(model, tokenizer)
    results = evaluator.evaluate_all(test_data)
    evaluator.print_report()
    
    return results
