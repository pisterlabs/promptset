from .wp import exec_wp, extract_proof_count_score, extract_proofs_and_goals
from langchain.evaluation.schema import StringEvaluator
from typing import Any, Optional


class AnnotationEvaluator(StringEvaluator):
    """
    Evaluate the output of the annotation chain
    """

    def _evaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any
    ) -> dict:
        program = prediction
        classification_counts = kwargs.get("classification_counts", {})
        headers_path = kwargs.get("headers_path", {})
        score = self.rank(program, classification_counts, headers_path)
        proved, goals = extract_proofs_and_goals(self.wp_result.stdout)

        return {
            "program": prediction,
            "rank": score,
            "classifications": classification_counts,
            "proved": proved,
            "goals": goals,
        }

    def rank(self, program: str, classification_counts: dict, headers_path: str):
        """
        ranks an annotated program with a value from 0 to 1
        scoring is based on the following criteria:

        Main goals:
        all annotations proved -> +0.75
        half annotations proved -> +0.55
        correct syntax -> +0.25


        Extras:
        requires -> +0.02
        assigns -> +0.015
        loop invariant -> +0.01
        ensures -> +0.01
        other -> +0.005

        """
        try:
            self.wp_result = exec_wp(annotated_program=program, headers_path=headers_path)
            if self.wp_result.returncode != 0:
                wp_score = 0
            else:
                wp_score = extract_proof_count_score(self.wp_result.stdout)

            classification_score = get_classifications_score(classification_counts)

            return wp_score + classification_score
        except:
            return -1, -1, 0


def get_classifications_score(classification_counts):
    classification_values = {
        "requires": 0.02,
        "assigns": 0.015,
        "loop invariant": 0.01,
        "ensures": 0.01,
    }
    return sum(
        classification_counts[key] * classification_values.get(key, 0.005)
        for key in classification_counts
    )
