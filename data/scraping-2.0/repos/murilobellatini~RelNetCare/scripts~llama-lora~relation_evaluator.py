from tqdm import tqdm
import openai
import json
from collections import defaultdict, OrderedDict

class RelationExtractorEvaluator:
    """Evaluate the performance of a relation extraction model on a test dataset.
    
    Attributes:
        model (str): Model name to use for inference.
        openai_api (str): Base API URL for the OpenAI model.
        api_key (str): API key for accessing the model.
    """

    def __init__(self):
        self.api_key = "EMPTY"
        self.api_base = "http://localhost:8000/v1"
        self.model = "vicuna-7b-v1.1"
        # Initialize OpenAI
        openai.api_key = self.api_key
        openai.api_base = self.api_base

    @staticmethod
    def _calculate_metrics_for_entry(true_labels, predicted_labels):
        """Calculate precision, recall, and F1 score for a given set of true and predicted labels."""
        if not true_labels and not predicted_labels:  # If both are empty
            return 1, 1, 1

        if not true_labels or not predicted_labels:  # If one of them is empty
            return 0, 0, 0

        true_set = set(true_labels)
        predicted_set = set(predicted_labels)

        tp = len(true_set & predicted_set)
        fp = len(predicted_set - true_set)
        fn = len(true_set - predicted_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    def infer_from_model(self, dialogue, preprompt=""):
        """Perform model inference given a dialogue and preprompt."""
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": preprompt + dialogue
            }]
        )
        return completion.choices[0].message.content

    def assess_performance_on_test_dataset(self, test_file_path, cap_size=None, return_details=False):
        """Evaluate the model's performance on the provided test dataset."""
        with open(test_file_path, 'r', encoding='utf8') as fp:
            test_data = json.load(fp)

        if cap_size:
            test_data = test_data[:cap_size]

        details = []
        results_per_class = defaultdict(list)
        pbar = tqdm(test_data, desc="Processing", dynamic_ncols=True, leave=False)

        overall_predictions = []
        overall_true = []

        for entry in pbar:
            prompt = "\n".join([message["value"] for message in entry["conversations"] if message["from"] == "human"])
            try:
                predicted_relations = json.loads(self.infer_from_model(prompt), object_pairs_hook=OrderedDict)
                true_relations = json.loads(entry["conversations"][1]["value"], object_pairs_hook=OrderedDict)

                predicted_labels = [str(pred_relation) for pred_relation in predicted_relations]
                true_labels = [str(true_relation) for true_relation in true_relations]

                for true_relation in true_relations:
                    results_per_class[true_relation.get('r')].append((predicted_labels, true_labels))

                precision, recall, f1 = self._calculate_metrics_for_entry(true_labels, predicted_labels)

                overall_predictions.extend(predicted_labels)
                overall_true.extend(true_labels)

                pbar.set_description(f"P: {precision:.2f} | R: {recall:.2f} | F1: {f1:.2f}")

                if return_details:
                    details.append({
                        "id": entry['id'],
                        "prompt": prompt,
                        "predicted_relations": predicted_relations,
                        "true_relations": true_relations,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1 
                    })

            except Exception as e:
                tqdm.write(f"Error processing entry with id {entry['id']}: {e}")

        overall_precision, overall_recall, overall_f1 = self._calculate_metrics_for_entry(overall_true, overall_predictions)

        per_class_results = {}
        for relation, labels_list in results_per_class.items():
            preds, trues = [], []
            for preds_labels, true_labels in labels_list:
                preds.extend(preds_labels)
                trues.extend(true_labels)

            precision, recall, f1 = self._calculate_metrics_for_entry(trues, preds)

            per_class_results[relation] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

        result = {
            "overall": {
                "precision": overall_precision,
                "recall": overall_recall,
                "f1": overall_f1
            },
            "per_class": per_class_results
        }

        if return_details:
            result["details"] = details

        return result

class FileManager:
    """Handle reading and writing of files."""
    
    @staticmethod
    def read_json_file(file_path):
        """Read a JSON file and return its content."""
        with open(file_path, 'r', encoding='utf8') as fp:
            return json.load(fp)
    
    # Any other file-related functions can be added here

if __name__ == "__main__":
    evaluator = RelationExtractorEvaluator()

    # Define path for the test dataset
    test_file_path = "/home/murilo/RelNetCare/data/processed/dialog-re-llama-11cls-2spkr/dialog-re-llama-11cls-2spkr-test.json"

    # Evaluate performance
    results = evaluator.assess_performance_on_test_dataset(test_file_path, cap_size=None, return_details=True)
    print(results['overall'])
