import unittest
import csv
from langchain_sentence_transformers import match_questions_ensemble

realisticQuestions = [
    "What was the last interaction with Oscar Mitchell?",
    "When was the last time we interacted with Ella Thompson?",
    "When was the last interaction with Larian Studios, and by whom?",
    "What do previous wins look like in BeanBrew?",
    "What leads are being worked on by SDR Lucas Warren?",
    "What contacts are being worked on by SDR Lily Turner?",
]

tweakedQuestionsWithGrammarMistakes = [
    "what was the last interaction we had with Mike Smith?",
    "when was the most recent interaction with Mike Smith?",
    "who was the last one interacted with Uber?",
    "descirbe the previous wins for Nike?",
    "on what leads does Mike Smith working?",
    "on what contact's Mike Smith do work?",
]


class TestQuestionMatching(unittest.TestCase):

    def setUp(self):
        self.csv_file = open("src/mixed_libraries/mismatches_tweaked_langchain_openai.csv", "w", newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["User Input", "Top Matched Question", "2nd Matched Question", "3rd Matched Question"])

    def test_matching(self):
        full_matches = 0

        for idx, user_input in enumerate(tweakedQuestionsWithGrammarMistakes):
            print(f"\nMatching for input: {user_input}")
            results = match_questions_ensemble(realisticQuestions, user_input)
            self.assertTrue(results)  # Ensure there's always a result

            top_match_question = results[0][0]
            expected_question = realisticQuestions[idx]

            # Check similarity
            if results[0][1] < 0.65:
                # Extracting similarity scores for the top three (or fewer) results
                similarity_scores = [res[1] for res in results[:3]]

                # Create a list for the CSV row
                csv_row = [user_input, f'"{results[0][0]}" with score {similarity_scores[0]:.3f}']

                # Append similarity scores for the 2nd and 3rd results, if they exist. Otherwise, append "N/A".
                for score in similarity_scores[1:]:
                    csv_row.append(f'Score: {score:.3f}')
                while len(csv_row) < 4:  # Ensure the row has 4 columns
                    csv_row.append("N/A")

                # Write to CSV
                self.csv_writer.writerow(csv_row)
            elif top_match_question != expected_question:
                # Writing mismatch data to CSV
                self.csv_writer.writerow([user_input,
                                          results[0][0] if len(results) > 0 else "N/A",
                                          results[1][0] if len(results) > 1 else "N/A",
                                          results[2][0] if len(results) > 2 else "N/A"])
            else:
                full_matches += 1

        print(f"\nTotal full matches: {full_matches}/{len(tweakedQuestionsWithGrammarMistakes)}")

        # Calculate and print the percentage
        match_percentage = (full_matches / len(tweakedQuestionsWithGrammarMistakes)) * 100
        print(f"Match percentage: {match_percentage:.2f}%")

    def tearDown(self):
        self.csv_file.close()


if __name__ == "__main__":
    unittest.main()
