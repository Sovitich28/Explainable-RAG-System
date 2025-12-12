"""
Evaluation Module
Test suite and metrics for evaluating the RAG system.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from typing import Dict, List

from rouge_score import rouge_scorer

from rag_system import ExplainableRAGSystem


class RAGEvaluator:
    """Evaluate the RAG system with various metrics."""

    def __init__(self, rag_system: ExplainableRAGSystem):
        """
        Initialize the evaluator.

        Args:
            rag_system: Initialized RAG system
        """
        self.rag_system = rag_system
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def create_test_set(self) -> List[Dict[str, any]]:
        """
        Create a test set of questions with gold-standard answers.

        Returns:
            List of test cases
        """
        test_cases = [
            {
                "question": "What is the EU renewable energy target for 2030?",
                "gold_answer": "The EU has set a binding target to achieve at least 42.5% renewable energy in final energy consumption by 2030, with an aspirational goal of 45%.",
                "expected_entities": ["Target", "Policy"],
                "expected_relations": ["MANDATES", "SET_BY"],
            },
            {
                "question": "What does RED III stand for and when was it adopted?",
                "gold_answer": "RED III stands for Renewable Energy Directive III, which was adopted in 2023.",
                "expected_entities": ["Legislation"],
                "expected_relations": ["IMPLEMENTS"],
            },
            {
                "question": "Which renewable energy sources are promoted by the Green Deal?",
                "gold_answer": "The Green Deal promotes solar energy, wind energy (both onshore and offshore), hydroelectric power, and biomass/biogas with sustainability criteria.",
                "expected_entities": ["Policy", "RenewableSource"],
                "expected_relations": ["PROMOTES"],
            },
            {
                "question": "What is the renewable energy target for the transport sector by 2030?",
                "gold_answer": "Member states must ensure that renewable energy accounts for at least 29% of final energy consumption in the transport sector by 2030.",
                "expected_entities": ["Target", "Sector"],
                "expected_relations": ["IMPACTS", "APPLIES_TO"],
            },
            {
                "question": "Which countries are mentioned in relation to renewable energy implementation?",
                "gold_answer": "Germany, France, Spain, Italy, Poland, Austria, and Sweden are mentioned as EU member states implementing renewable energy policies.",
                "expected_entities": ["Country"],
                "expected_relations": ["APPLIES_TO", "SUPPORTS"],
            },
            {
                "question": "What is the Fit for 55 package?",
                "gold_answer": "The Fit for 55 package is a comprehensive set of legislative proposals presented in July 2021 to align EU policies with the 2030 climate target of reducing greenhouse gas emissions by at least 55% compared to 1990 levels.",
                "expected_entities": ["Policy", "Legislation"],
                "expected_relations": ["MANDATES"],
            },
            {
                "question": "What is Germany's renewable electricity target for 2030?",
                "gold_answer": "Germany has committed to achieving 80% renewable electricity by 2030, exceeding the EU-wide target.",
                "expected_entities": ["Country", "Target"],
                "expected_relations": ["APPLIES_TO"],
            },
            {
                "question": "How does the Green Deal impact the buildings sector?",
                "gold_answer": "The buildings sector is targeted through the Energy Performance of Buildings Directive, requiring all new buildings to be zero-emission by 2030 and existing buildings to undergo deep renovations.",
                "expected_entities": ["Policy", "Sector"],
                "expected_relations": ["IMPACTS"],
            },
        ]

        return test_cases

    def calculate_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores.

        Args:
            generated: Generated answer
            reference: Gold-standard answer

        Returns:
            Dictionary of ROUGE scores
        """
        scores = self.rouge_scorer.score(reference, generated)

        return {
            "rouge1_f": scores["rouge1"].fmeasure,
            "rouge2_f": scores["rouge2"].fmeasure,
            "rougeL_f": scores["rougeL"].fmeasure,
        }

    def evaluate_faithfulness(self, answer: str, citations: str) -> float:
        """
        Evaluate if the answer is faithful to the cited sources.
        Simple heuristic: check if citations are present and non-empty.

        Args:
            answer: Generated answer
            citations: Citations provided

        Returns:
            Faithfulness score (0-1)
        """
        # Simple heuristic: citations should be present and substantial
        if not citations or len(citations.strip()) < 20:
            return 0.0

        # Check if answer references sources
        if "source" in citations.lower() or "chunk" in citations.lower():
            return 1.0

        return 0.5

    def evaluate_graph_usage(
        self,
        graph_data: Dict,
        expected_entities: List[str],
        expected_relations: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate if the correct graph entities and relationships were retrieved.

        Args:
            graph_data: Retrieved graph data
            expected_entities: Expected entity types
            expected_relations: Expected relationship types

        Returns:
            Dictionary with entity and relation recall scores
        """
        retrieved_entities = set()
        retrieved_relations = set()

        # Extract entity types
        for node in graph_data.get("nodes", []):
            retrieved_entities.update(node.get("labels", []))

        # Extract relationship types
        for rel in graph_data.get("relationships", []):
            retrieved_relations.add(rel.get("type", ""))

        # Calculate recall
        entity_recall = (
            len(set(expected_entities) & retrieved_entities) / len(expected_entities)
            if expected_entities
            else 0
        )
        relation_recall = (
            len(set(expected_relations) & retrieved_relations) / len(expected_relations)
            if expected_relations
            else 0
        )

        return {"entity_recall": entity_recall, "relation_recall": relation_recall}

    def run_evaluation(self, test_cases: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Run full evaluation on test cases.

        Args:
            test_cases: List of test cases

        Returns:
            Evaluation results with metrics
        """
        results = []

        for i, test_case in enumerate(test_cases):
            print(
                f"Evaluating test case {i+1}/{len(test_cases)}: {test_case['question']}"
            )

            # Query the system
            response = self.rag_system.query(test_case["question"])

            # Calculate metrics
            rouge_scores = self.calculate_rouge(
                response["answer"], test_case["gold_answer"]
            )
            faithfulness = self.evaluate_faithfulness(
                response["answer"], response["citations"]
            )
            graph_metrics = self.evaluate_graph_usage(
                response["graph_data"],
                test_case["expected_entities"],
                test_case["expected_relations"],
            )

            # Store results
            result = {
                "question": test_case["question"],
                "generated_answer": response["answer"],
                "gold_answer": test_case["gold_answer"],
                "rouge_scores": rouge_scores,
                "faithfulness": faithfulness,
                "graph_metrics": graph_metrics,
                "explanation": response["explanation"],
                "citations": response["citations"],
            }

            results.append(result)

        # Calculate aggregate metrics
        avg_rouge1 = sum(r["rouge_scores"]["rouge1_f"] for r in results) / len(results)
        avg_rouge2 = sum(r["rouge_scores"]["rouge2_f"] for r in results) / len(results)
        avg_rougeL = sum(r["rouge_scores"]["rougeL_f"] for r in results) / len(results)
        avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
        avg_entity_recall = sum(
            r["graph_metrics"]["entity_recall"] for r in results
        ) / len(results)
        avg_relation_recall = sum(
            r["graph_metrics"]["relation_recall"] for r in results
        ) / len(results)

        summary = {
            "total_test_cases": len(results),
            "average_metrics": {
                "rouge1_f": avg_rouge1,
                "rouge2_f": avg_rouge2,
                "rougeL_f": avg_rougeL,
                "faithfulness": avg_faithfulness,
                "entity_recall": avg_entity_recall,
                "relation_recall": avg_relation_recall,
            },
            "detailed_results": results,
        }

        return summary

    def generate_report(self, evaluation_results: Dict[str, any], output_file: str):
        """
        Generate a detailed evaluation report.

        Args:
            evaluation_results: Results from run_evaluation
            output_file: Path to save the report
        """
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# RAG System Evaluation Report\n\n")

            # Summary
            f.write("## Summary\n\n")
            f.write(f"Total test cases: {evaluation_results['total_test_cases']}\n\n")

            f.write("### Average Metrics\n\n")
            metrics = evaluation_results["average_metrics"]
            f.write(f"- **ROUGE-1 F1**: {metrics['rouge1_f']:.3f}\n")
            f.write(f"- **ROUGE-2 F1**: {metrics['rouge2_f']:.3f}\n")
            f.write(f"- **ROUGE-L F1**: {metrics['rougeL_f']:.3f}\n")
            f.write(f"- **Faithfulness**: {metrics['faithfulness']:.3f}\n")
            f.write(f"- **Entity Recall**: {metrics['entity_recall']:.3f}\n")
            f.write(f"- **Relation Recall**: {metrics['relation_recall']:.3f}\n\n")

            # Detailed results
            f.write("## Detailed Results\n\n")

            for i, result in enumerate(evaluation_results["detailed_results"], 1):
                f.write(f"### Test Case {i}\n\n")
                f.write(f"**Question**: {result['question']}\n\n")
                f.write(f"**Gold Answer**: {result['gold_answer']}\n\n")
                f.write(f"**Generated Answer**: {result['generated_answer']}\n\n")

                f.write("**Metrics**:\n")
                f.write(f"- ROUGE-1: {result['rouge_scores']['rouge1_f']:.3f}\n")
                f.write(f"- ROUGE-2: {result['rouge_scores']['rouge2_f']:.3f}\n")
                f.write(f"- ROUGE-L: {result['rouge_scores']['rougeL_f']:.3f}\n")
                f.write(f"- Faithfulness: {result['faithfulness']:.3f}\n")
                f.write(
                    f"- Entity Recall: {result['graph_metrics']['entity_recall']:.3f}\n"
                )
                f.write(
                    f"- Relation Recall: {result['graph_metrics']['relation_recall']:.3f}\n\n"
                )

                f.write(f"**Explanation**: {result['explanation']}\n\n")
                f.write(f"**Citations**: {result['citations']}\n\n")
                f.write("---\n\n")

        print(f"Report saved to {output_file}")


def main():
    """Main execution function."""
    # Initialize RAG system
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    rag = ExplainableRAGSystem(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # Load chunks
    from data_preparation import main as prepare_data

    chunks = prepare_data()
    rag.load_and_index_chunks(chunks)

    # Create evaluator
    evaluator = RAGEvaluator(rag)

    # Create test set
    test_cases = evaluator.create_test_set()

    # Run evaluation (limit to 3 cases for demo)
    results = evaluator.run_evaluation(test_cases[:3])

    # Generate report
    base_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(base_dir, "evaluation_report.md")
    evaluator.generate_report(results, report_path)

    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(json.dumps(results["average_metrics"], indent=2))

    rag.close()


if __name__ == "__main__":
    main()
