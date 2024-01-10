import unittest
import csv
from langchain_sentence_transformers import match_questions_ensemble, questions, user_inputs

class TestQuestionMatching(unittest.TestCase):

    def setUp(self):
        self.csv_file = open("src/mixed_libraries/mismatches_ensemble_mixed_2_models.csv", "w", newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # Writing the headers for the CSV file
        self.csv_writer.writerow(["User Input", "Top Matched Question", "2nd Matched Question", "3rd Matched Question"])

    def test_matching(self):
        full_matches = 0

        for idx, user_input in enumerate(user_inputs):
            print(f"\nMatching for input: {user_input}")
            results = match_questions_ensemble(questions, user_input)
            self.assertTrue(results)  # Ensure there's always a result

            top_match_question = results[0][0]
            expected_question = questions[idx]

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

        print(f"\nTotal full matches: {full_matches}/{len(user_inputs)}")

        # Calculate and print the percentage
        match_percentage = (full_matches / len(user_inputs)) * 100
        print(f"Match percentage: {match_percentage:.2f}%")

    def tearDown(self):
        # Closing the CSV file after all tests complete
        self.csv_file.close()

if __name__ == "__main__":
    unittest.main()
