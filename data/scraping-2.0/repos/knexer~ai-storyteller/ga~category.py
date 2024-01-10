import openai

import random
import re


class Category:
    def __init__(self, conditioning_info, premise, notes):
        self.conditioning_info = conditioning_info
        self.premise = premise
        self.notes = notes

    def category_name(self):
        raise NotImplementedError("Derived classes must define a category name")

    def best_possible_score(self):
        raise NotImplementedError("Derived classes must define the best possible score")

    def rubric(self):
        raise NotImplementedError("Derived classes must define the rubric")

    def recommendations_reminder(self):
        raise NotImplementedError(
            "Derived classes must define the recommendations reminder"
        )

    def scoring_prompt(self):
        return f"""I'm working on an illustrated children's story for a client.
They gave me a premise:
{self.premise}
They gave me other requirements:
{self.conditioning_info}

I have elaborated on the premise, producing these notes:
{self.notes}

Critique this story's {self.category_name()} based on this rubric.

{self.rubric()}

Follow the list structure of the rubric. For each item, discuss things the story's {self.category_name()} do well and things they do poorly in that respect, then lastly score that item out of 5. Be as harsh as possible!

End your review with "Overall Score: <sum of item scores>/{self.best_possible_score()}". N/A for any item should count as a 5."""

    def score(self, verbose=False, n=1):
        self.scores = []
        prompt = self.scoring_prompt()
        if verbose:
            print(prompt)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt},
            ],
            n=n,
            temperature=1,
        )

        if verbose:
            [print(choice.message.content) for choice in response.choices]

        scores = [
            self.parse_score(prompt, choice.message.content)
            for choice in response.choices
        ]
        scores = [score for score in scores if score is not None]

        actual = len(scores)
        if actual < n:
            print(
                f"WARNING: Only {actual} scores could be parsed of the {n} responses."
            )
            self.score(verbose, n - actual)

        self.scores.extend(scores)

    def is_scored(self):
        return hasattr(self, "scores")

    def average_score(self):
        return sum([int(score["score"]) for score in self.scores]) / len(self.scores)

    def normalized_score(self):
        return self.average_score() / self.best_possible_score()

    def parse_score(self, prompt, response):
        score_regex = re.compile(
            r"Overall Score:\s*(\d{1,2}(?:\.\d{1,2})?)/"
            + f"{self.best_possible_score()}"
        )
        match = score_regex.search(response)

        if not match:
            print(f"WARNING: Could not parse score from response: {response}")
            return None

        overall_score = float(match.group(1))

        return {
            "conversation": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "score": overall_score,
        }

    def recommendations_prompt(self):
        return f"""Based on your feedback, give three independent, detailed recommendations for how to improve the outline.
Each recommendation should solve a problem highlighted above and specify every detail on what should be changed and how. DO NOT list multiple alternatives, options, or examples; give one detailed, concrete solution.
Each recommendation should be independent of the others, as they will be evaluated separately.

{self.recommendations_reminder()}

Give your recommendations in a numbered list format. Omit preface, omit a summary, and omit other notes; include only the list itself."""

    def make_recommendation(self, verbose=False):
        # Choose a score to improve on, weighted by mismatch between score and best possible score
        num_missing_points = self.best_possible_score() * len(self.scores) - sum(
            [score["score"] for score in self.scores]
        )
        missing_point = random.uniform(0, num_missing_points)
        current = 0
        for score in self.scores:
            current += self.best_possible_score() - score["score"]
            if current > missing_point:
                return self.make_recommendation_from_score(score, verbose)

    def make_recommendation_from_score(self, score, verbose=False):
        prompt = self.recommendations_prompt()
        if verbose:
            print(prompt)
        recommendations = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=score["conversation"] + [{"role": "user", "content": prompt}],
            n=1,
            temperature=1,
        )
        if verbose:
            print(recommendations.choices[0].message.content)
        pick_best_prompt = "Which of those is the best recommendation? Repeat the recommendation, without the number and without any other preface text."
        best_recommendation = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=score["conversation"]
            + [{"role": "user", "content": prompt}]
            + [recommendations.choices[0].message]
            + [{"role": "user", "content": pick_best_prompt}],
            n=1,
            temperature=1,
        )
        if verbose:
            print(best_recommendation.choices[0].message.content)

        return best_recommendation.choices[0].message.content

    def apply_recommendation_prompt(self, recommendation):
        return f"""I'm working on an illustrated children's story for a client.
They gave me a premise:
{self.premise}
They gave me other requirements:
{self.conditioning_info}

I have elaborated on the premise, producing these notes:
====== ORIGINAL NOTES ======
{self.notes}
====== END ORIGINAL NOTES ======

I received this feedback on the story's {self.category_name()}:
{recommendation}

Revise the notes to incorporate this feedback.
Begin your response by strategizing how you will change the story notes based on the feedback. In addition to an overall plan, also carefully identify and resolve ambiguity in the recommendation. How are other aspects of the story impacted by the recommendation? Is there anything the recommendation doesn't specify adequately? Resolve that ambiguity before starting the revised notes, to ensure they are self-consistent, specific and fully meet the recommendation's objective.
Then write your revised notes, wrapped in "====== REVISED NOTES ======" and "====== END REVISED NOTES ======"."""
