from datasets import Dataset, DatasetDict
import shutil
import os
from sklearn.metrics import f1_score, precision_score, recall_score
from matplotlib.lines import Line2D
import json
import openai
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns        
from ast import literal_eval
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
from src.paths import LOCAL_DATA_PATH

from src.utils import get_value_from_locals
from src.utils import fix_cls_metrics_dump
from src.config import LLMTransformationConfig
import warnings

def safe_divide(numerator, denominator):
    if denominator == 0:
        warnings.warn(f"Attempted division by zero. Returning None instead.")
        return None
    return numerator / denominator

config = LLMTransformationConfig()

class RelationExtractorEvaluator:
    """Evaluate the performance of a relation extraction model on a test dataset.
    
    Attributes:
        model (str): Model name to use for inference.
        openai_api (str): Base API URL for the OpenAI model.
        api_key (str): API key for accessing the model.
    """

    def __init__(self, config=config):
        self.api_key = "EMPTY"
        self.api_base = "http://localhost:8000/v1"
        self.model = "vicuna-7b-v1.1"
        self.config = config
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

    def infer_from_model(self, dialogue, preprompt="", max_output_length=512):
        """Perform model inference given a dialogue and preprompt."""
        completion = openai.ChatCompletion.create(
            model=self.model,
            temperature=0,
            max_tokens=max_output_length,  # Set your desired max length
            messages=[{
                "role": "user",
                "content": preprompt + dialogue
            }]
        )
        return completion.choices[0].message.content

    def assess_performance_on_test_dataset(self, test_file_path, cap_size=None, return_details=False, remove_ordereddict=True, export_report=True):
        """Evaluate the model's performance on the provided test dataset."""
        with open(test_file_path, 'r', encoding='utf8') as fp:
            test_data = json.load(fp)

        if cap_size:
            test_data = test_data[:cap_size]

        details = []
        results_per_class = defaultdict(list)
        pbar = tqdm(test_data, dynamic_ncols=True, leave=False, position=0)

        overall_predictions = []
        overall_true = []

        result_df = pd.DataFrame(columns=["id","prompt","dialogue","true_labels","raw_inference","predicted_labels","correct_labels","wrong_labels", "missing_labels","f1s","precision","recall","error_message"])
        errors = []
        
        total_precision, total_recall, total_f1 = 0, 0, 0
        processed_entries = 0
        errors_count = 0


        for entry in pbar:
            precision = None 
            recall = None 
            f1 = None
            dialogue = None
            true_labels = None
            raw_inference = None
            predicted_labels = None
            correct_labels = None
            wrong_labels = None
            avg_precision = None
            avg_recall = None
            avg_f1 = None
            
            prompt = "\n".join([message["value"] for message in entry["conversations"] if message["from"] == "human"])
            try:
                raw_inference = self.infer_from_model(prompt)
                if self.config.cls_task_only:
                    predicted_relations = [raw_inference]
                    true_relations = [entry["conversations"][1]["value"]]
                else:
                    predicted_relations = json.loads(raw_inference, object_pairs_hook=OrderedDict) if not raw_inference in ['', '\n'] else []
                    true_relations = json.loads(entry["conversations"][1]["value"], object_pairs_hook=OrderedDict)

                predicted_labels = [str(pred_relation) for pred_relation in predicted_relations]
                true_labels = [str(true_relation) for true_relation in true_relations]

                if not self.config.cls_task_only:
                    for true_relation in true_relations:
                        results_per_class[true_relation.get('r')].append((predicted_labels, true_labels))

                precision, recall, f1 = self._calculate_metrics_for_entry(true_labels, predicted_labels)

                total_precision += precision
                total_recall += recall
                total_f1 += f1
                processed_entries += 1

                overall_predictions.extend(predicted_labels)
                overall_true.extend(true_labels)

                avg_precision = total_precision / processed_entries
                avg_recall = total_recall / processed_entries
                avg_f1 = total_f1 / processed_entries
        
                pbar.set_description(f"Avg P: {avg_precision:.1%} | Avg R: {avg_recall:.1%} | Avg F1: {avg_f1:.1%} | Errors: {errors_count}/{len(test_data)} ({errors_count/len(test_data):.0%})")

                # Calculate correct and wrong labels
                correct_labels = list(set(true_labels) & set(predicted_labels)) # true_positives
                wrong_labels = list(set(predicted_labels) - set(true_labels)) # false_positives
                missing_labels = list(set(true_labels) - set(predicted_labels)) # false_negatives
                
                dialogue = prompt.split('Input:')[-1].replace('Output:','')

                if return_details:
                    detail = {
                        "id": entry['id'],
                        "prompt": prompt.replace(self.config.preprompt, ''),
                        "dialogue": dialogue,
                        "true_labels": true_labels,
                        "raw_inference": raw_inference,
                        "predicted_labels": predicted_labels,
                        "correct_labels": correct_labels,
                        "wrong_labels": wrong_labels,
                        "missing_labels": missing_labels,
                        "f1s": f1,
                        "precision": precision,
                        "recall": recall,
                        "error_message": ''

                    }
                    details.append(detail)
                    result_df.loc[len(result_df.index)] = detail
    

            except Exception as e:
                errors_count += 1

                errors.append(f"{entry['id']}: {e}")

                local_vars = locals()  # Capture local variables where the function will be called

                # Inside your loop
                error_detail = {
                    "id": entry['id'],
                    "prompt": get_value_from_locals('prompt', local_vars),
                    "dialogue": get_value_from_locals('dialogue', local_vars),
                    "true_labels": get_value_from_locals('true_relations', local_vars),
                    "raw_inference": get_value_from_locals('raw_inference', local_vars),
                    "predicted_labels": get_value_from_locals('predicted_relations', local_vars),
                    "correct_labels": get_value_from_locals('correct_labels', local_vars),
                    "wrong_labels": get_value_from_locals('wrong_labels', local_vars),
                    "missing_labels": get_value_from_locals('missing_labels', local_vars),
                    "f1s": get_value_from_locals('f1', local_vars, default_value=0),
                    "precision": get_value_from_locals('precision', local_vars, default_value=0),
                    "recall": get_value_from_locals('recall', local_vars, default_value=0),
                    "error_message": str(e)
                }

                result_df.loc[len(result_df.index)] = error_detail
                


        output_path = test_file_path.replace('.json', '')
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not self.config.cls_task_only:
            if remove_ordereddict:
                for col in result_df.columns:
                    if 'labels' in col:
                        result_df[col] = result_df[col].apply(lambda x: [xi.replace('OrderedDict', '') for xi in x] if x is not None else None)
                        result_df[col] = result_df[col].apply(lambda labels: json.dumps([dict(literal_eval(x)) for x in labels], indent=1) if labels is not None else None)

                
        output_path = f"{output_path}_{current_time}.xlsx"
        
        reordered_cols = ['id','prompt','raw_inference', 'true_labels', 'predicted_labels', 'correct_labels','wrong_labels','missing_labels','dialogue','f1s','precision','recall','error_message']
        result_df = result_df[reordered_cols]
        for c in ['f1s', 'precision', 'recall']:
            result_df[c] = result_df[c].fillna(0)
        if export_report:
            result_df.to_excel(output_path, index=False)
        print(f"\nScript successfully executed!")
        if all(var is not None for var in [avg_precision, avg_recall, avg_f1]):
            print(f"Avg P: {avg_precision:.1%} | Avg R: {avg_recall:.1%} | Avg F1: {avg_f1:.1%} | Errors: {errors_count}/{len(test_data)} ({errors_count/len(test_data):.0%})")
        else:
            print("Couldn't calculate metrics. Some variables are not populated.")
        print(f"# INFERENCE REPORT\n{output_path}\n")

            
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

        return result_df

    def assess_performance_on_lists(self, true_labels_list, pred_labels_list, output_path=None, cap_size=None, return_details=True, remove_ordereddict=True):
        """Evaluate the model's performance on provided true and predicted label lists."""
        
        details = []
        results_per_class = defaultdict(list)
        
        overall_predictions = []
        overall_true = []

        result_df = pd.DataFrame(columns=["id","prompt","dialogue","true_labels","raw_inference","predicted_labels","correct_labels","wrong_labels", "missing_labels","f1s","precision","recall","error_message"])
        errors = []
        
        total_precision, total_recall, total_f1 = 0, 0, 0
        processed_entries = 0
        errors_count = 0

        for true_labels, predicted_labels in zip(true_labels_list, pred_labels_list):
            precision = None 
            recall = None 
            f1 = None
            dialogue = None
            raw_inference = None

            if cap_size and processed_entries >= cap_size:
                break

            try:
                if not self.config.cls_task_only:
                    for true_relation in true_labels:
                        results_per_class[true_relation.get('relation', true_relation.get('r'))].append((predicted_labels, true_labels))

                predicted_labels = [str(pred_relation) for pred_relation in predicted_labels]
                true_labels = [str(true_relation) for true_relation in true_labels]
                
                precision, recall, f1 = self._calculate_metrics_for_entry(true_labels, predicted_labels)

                total_precision += precision
                total_recall += recall
                total_f1 += f1
                processed_entries += 1

                overall_predictions.extend(predicted_labels)
                overall_true.extend(true_labels)

                avg_precision = total_precision / processed_entries
                avg_recall = total_recall / processed_entries
                avg_f1 = total_f1 / processed_entries

                # Calculate correct and wrong labels
                correct_labels = list(set(true_labels) & set(predicted_labels)) # true_positives
                wrong_labels = list(set(predicted_labels) - set(true_labels)) # false_positives
                missing_labels = list(set(true_labels) - set(predicted_labels)) # false_negatives

                if return_details:
                    detail = {
                        "id": processed_entries,  # You might want to change this to something meaningful
                        "prompt": "",  # Add your prompt logic here
                        "dialogue": dialogue,
                        "true_labels": true_labels,
                        "raw_inference": raw_inference,
                        "predicted_labels": predicted_labels,
                        "correct_labels": correct_labels,
                        "wrong_labels": wrong_labels,
                        "missing_labels": missing_labels,
                        "f1s": f1,
                        "precision": precision,
                        "recall": recall,
                        "error_message": ''
                    }
                    details.append(detail)
                    result_df.loc[len(result_df.index)] = detail

            except Exception as e:
                errors_count += 1
                error = f"Entry {processed_entries}: {e}"
                errors.append(error)
                print(error)
                

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not self.config.cls_task_only:
            if remove_ordereddict:
                for col in result_df.columns:
                    if 'labels' in col:
                        result_df[col] = result_df[col].apply(lambda x: [xi.replace('OrderedDict', '') for xi in x] if x is not None else None)
                        result_df[col] = result_df[col].apply(lambda labels: json.dumps([dict(literal_eval(x)) for x in labels], indent=1) if labels is not None else None)

        if output_path:
            output_path = f"{output_path}_{current_time}.xlsx"
            
            reordered_cols = ['id','prompt','raw_inference', 'true_labels', 'predicted_labels', 'correct_labels','wrong_labels','missing_labels','dialogue','f1s','precision','recall','error_message']
            result_df = result_df[reordered_cols]
            for c in ['f1s', 'precision', 'recall']:
                result_df[c] = result_df[c].fillna(0)
            result_df.to_excel(output_path, index=False)
            print(f"\nScript successfully executed!")
            if all(var is not None for var in [avg_precision, avg_recall, avg_f1]):
                pass
            else:
                print("Couldn't calculate metrics. Some variables are not populated.")
        else:
            print('Not output_path provided, nothing dump!')    
        
        print(f"# INFERENCE REPORT\n{output_path}\n")

            
        overall_precision, overall_recall, overall_f1 = self._calculate_metrics_for_entry(overall_true, overall_predictions)

        per_class_results = {}
        for relation, labels_list in results_per_class.items():
            preds, trues = [], []
            for preds_labels, true_labels in labels_list:
                preds.extend(preds_labels)
                trues.extend(true_labels)

            precision, recall, f1 = self._calculate_metrics_for_entry(str(trues), str(preds))

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

        return result_df


class FileManager:
    """Handle reading and writing of files."""
    
    @staticmethod
    def read_json_file(file_path):
        """Read a JSON file and return its content."""
        with open(file_path, 'r', encoding='utf8') as fp:
            return json.load(fp)
    
    # Any other file-related functions can be added here

class RelationGranularMetrics(RelationExtractorEvaluator):
    def __init__(self, df, ontology):
        self.df = df
        self.ontology = ontology
        self.result = {}
    
    def aggregate_by_relation(self, group):
        metrics_by_relation = {}
        all_true_labels = literal_eval(str(group['true_labels'].iloc[0])) if any(group['true_labels']) else ["Null-Relation"]
        all_predicted_labels = literal_eval(str(group['predicted_labels'].iloc[0])) if any(group['predicted_labels']) else ["Null-Relation"]
        
        if '[]' in str(all_true_labels):
            all_true_labels =  ["Null-Relation"]
        if '[]' in str(all_predicted_labels):
            all_predicted_labels =  ["Null-Relation"]
        
        for r in self.ontology:
            true_labels = [str(x) for x in all_true_labels]
            predicted_labels = [str(x) for x in all_predicted_labels if r in str(x)]
            
            if not true_labels and not predicted_labels:
                metrics_by_relation[r] = {'precision': None, 'recall': None, 'f1': None}
            else:
                precision, recall, f1 = self._calculate_metrics_for_entry(true_labels, predicted_labels)
                metrics_by_relation[r] = {'precision': precision, 'recall': recall, 'f1': f1}
        
        return metrics_by_relation
    
    def process(self):
        grouped = self.df.groupby('id')
        for name, group in grouped:
            self.result[name] = self.aggregate_by_relation(group)
        return self.result
    
    def to_dataframe(self):
        chart_df = pd.DataFrame.from_dict({(i, j): self.result[i][j] 
                                for i in self.result.keys() 
                                for j in self.result[i].keys()}, 
                                orient='index')
        return chart_df
    
    def plot_metrics(self, chart_df, figsize=(3,5)):
        agg_stats = chart_df.groupby(level=1).agg(['mean', 'std'])
        fig, ax = plt.subplots()
        agg_stats = agg_stats.sort_values(by=('f1', 'mean'), ascending=True)
        agg_stats.xs('mean', axis=1, level=1).plot(kind='barh', xerr=agg_stats.xs('std', axis=1, level=1), ax=ax, figsize=figsize)
        plt.xlabel('Metrics')
        plt.title('Average and Stddev for Relations')
        plt.xlim(-0.05, 1.05) 
        plt.legend(loc='lower right')
        plt.show()



        
class GranularMetricVisualizer:
    
    def __init__(self, df, model_name, test_dataset_stem, exp_group="", cls_task_only=False):
        
        def try_json_loads(data):
            try:
              return json.loads(data)
            except:
              return data
        
        dump_files = True
        metrics = ['f1', 'precision', 'recall']
        with_no_relation=True
        self.exp_group = exp_group
        
        df['true_labels'] = df.true_labels.apply(try_json_loads)
        df['predicted_labels'] = df.predicted_labels.apply(try_json_loads)
        
        # @TODO: assess the possibility of handling HALLUCINATED labels
        self.cls_task_only = cls_task_only
        if cls_task_only:
            relations = df.true_labels.explode().dropna().unique().tolist()
        else:
            relations = df.true_labels.apply(lambda rels: [r.get('relation', r.get('r')) for r in rels]).explode().dropna().unique().tolist()
        self.model_name = model_name
        self.test_dataset_stem = test_dataset_stem
        self.dump_files = dump_files
        self.with_no_relation = with_no_relation
        if with_no_relation:
            if 'null_relation' not in relations:
                relations.append('null_relation')
        self.relations = relations
        self.metrics = metrics
        test_dataset_stem = test_dataset_stem.replace('-prepBART','')
        self.data_stem = test_dataset_stem
        self.data_readme = LOCAL_DATA_PATH / f'processed/{test_dataset_stem}/README.md'
        self.dump_path = LOCAL_DATA_PATH / f"reports/{test_dataset_stem}/{model_name}"
        print(f"self.dump_path={self.dump_path}")
        if self.dump_files:
            self.dump_path.mkdir(parents=True, exist_ok=True)

        self.df = self.enrich_df(df)
        self.metrics_dict = self.extract_metrics(df)
        
    def enrich_df(self, df):
        df['failure_modes'] = df.apply(self.compute_failure_modes, relations=self.relations, cls_task_only=self.cls_task_only, axis=1)
        df[['precision', 'recall', 'f1s']] = df.apply(self.recompute_cls_metrics, axis=1)
        
        return df
        
    def extract_metrics(self, df):
        metrics = self.calculate_class_metrics(df['failure_modes'])
        metrics['per_class'] = self.calculate_per_class_metrics(df['failure_modes'])
        return metrics

    @staticmethod
    def recompute_cls_metrics(row):
        failure_modes = row['failure_modes']
        precision_list = []
        recall_list = []
        f1_list = []
        
        for rel, metrics in failure_modes.items():
            tp = metrics['counts']['tp']
            fp = metrics['counts']['fp']
            fn = metrics['counts']['fn']
            
            if tp + fp + fn == 0:
                continue  # Skip relations that have zero counts for everything

            precision = tp / (tp + fp) if tp + fp != 0 else 0
            recall = tp / (tp + fn) if tp + fn != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
            
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0
        avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0
        avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0

        return pd.Series({
            'precision': avg_precision,
            'recall': avg_recall,
            'f1s': avg_f1
        })

    # Assume metric_visualizer.df is your DataFrame
    # You should adjust 'failure_modes' to the correct column name if it's different
    @staticmethod
    def compute_failure_modes(row, relations, cls_task_only=False):
        output = {}
        true_label_dicts = row['true_labels']
        pred_label_dicts = row['predicted_labels']

        true_labels_str = [str(l) for l in true_label_dicts] if len(true_label_dicts) > 0 else ['null_relation']
        pred_labels_str = [str(l) for l in pred_label_dicts] if len(pred_label_dicts) > 0 else ['null_relation']

        for r in relations:
            true_labels_with_rel = [l for l in true_labels_str if r in l]
            pred_labels_with_rel = [l for l in pred_labels_str if r in l]

            tp_list = [l for l in true_labels_with_rel if l in pred_labels_with_rel]
            fp_list = [l for l in pred_labels_with_rel if l not in true_labels_with_rel]
            fn_list = [l for l in true_labels_with_rel if l not in pred_labels_with_rel]
            if not any(r in item for item in tp_list) and not any(r in item for item in fp_list):
                tn_list = [r]
            else:
                tn_list = []


            output[r] = {
                'list': {
                    'tp': tp_list,
                    'fp': fp_list,
                    'tn': tn_list,
                    'fn': fn_list
                },
                'counts': {
                    'tp': len(tp_list),
                    'fp': len(fp_list),
                    'tn': len(tn_list),
                    'fn': len(fn_list),
                }
            }
        return output

    @staticmethod
    def calculate_class_metrics(failure_modes_df):
        # Initialize sums for micro-average
        micro_sum = defaultdict(int)
        
        # Initialize lists for macro-average
        macro_precision = []
        macro_recall = []
        macro_f1 = []

        for row in failure_modes_df:
            for rel, data in row.items():
                tp = data['counts']['tp']
                fp = data['counts']['fp']
                fn = data['counts']['fn']
                
                if tp + fp + fn == 0:
                    continue
                # Sum counts for micro-average
                micro_sum['tp'] += tp
                micro_sum['fp'] += fp
                micro_sum['fn'] += fn
                
                # Calculate per-relation metrics
                if tp + fp == 0:
                    precision = 0
                else:
                    precision = tp / (tp + fp)
                
                if tp + fn == 0:
                    recall = 0
                else:
                    recall = tp / (tp + fn)
                
                if precision + recall == 0:
                    f1 = 0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)
                
                # Append to lists for macro-average
                macro_precision.append(precision)
                macro_recall.append(recall)
                macro_f1.append(f1)

        # Micro-average
        micro_precision = safe_divide(micro_sum['tp'], micro_sum['tp'] + micro_sum['fp'])
        micro_recall = safe_divide(micro_sum['tp'], micro_sum['tp'] + micro_sum['fn'])
        micro_f1 = safe_divide(2 * (micro_precision * micro_recall), micro_precision + micro_recall) if micro_precision is not None and micro_recall is not None else None

        # Macro-average
        macro_precision = safe_divide(sum(macro_precision), len(macro_precision))
        macro_recall = safe_divide(sum(macro_recall), len(macro_recall))
        macro_f1 = safe_divide(sum(macro_f1), len(macro_f1))

        return { 'micro_avg':
            {'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1},
            'macro_avg':
            {'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1}}

    @staticmethod
    def calculate_per_class_metrics(failure_modes_df):
        class_metrics = defaultdict(lambda: defaultdict(int))

        for row in failure_modes_df:
            for rel, data in row.items():
                tp = data['counts']['tp']
                fp = data['counts']['fp']
                fn = data['counts']['fn']
                
                # Update class-specific metrics
                class_metrics[rel]['tp'] += tp
                class_metrics[rel]['fp'] += fp
                class_metrics[rel]['fn'] += fn

        # Calculate the final metrics for each class
        final_class_metrics = {}
        for rel, counts in class_metrics.items():
            tp = counts['tp']
            fp = counts['fp']
            fn = counts['fn']
            
            # Skip if all counts are zero
            if tp + fp + fn == 0:
                continue
            
            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)
                
            if tp + fn == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)
                
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            final_class_metrics[rel] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        return final_class_metrics

    def visualize_class_metrics_distribution(self, df):
        df = df.rename({'f1s': 'f1'}, axis=1)
        # Create a long-form DataFrame suitable for sns.swarmplot
        df_melted = df.melt(value_vars=self.metrics)

        # Create the violinplot
        sns.violinplot(y='variable', x='value', data=df_melted, cut=0, inner=None, palette="coolwarm")

        # Overlay the stripplot
        sns.stripplot(y='variable', x='value', data=df_melted, color='black', size=5, alpha=0.3)

        # Calculate the mean and annotate it
        for idx, metric in enumerate(self.metrics):
            for type_idx, type_ in enumerate(('micro', 'macro')):
                mean_val = self.metrics_dict[f'{type_}_avg'][metric]
                color = 'darkblue' if type_ == 'micro' else 'blue'
                plt.scatter(mean_val, idx, color=color, s=100, zorder=3) 

                # Add a small vertical offset based on type_idx
                vertical_offset = idx + 0.1 + (0.1 * type_idx) 
                
                if mean_val is None:
                    warnings.warn(f"Mean value is None for {type_}. Skipping plotting for this value.")
                    plt.text(0, vertical_offset, f'{type_} mean: None',
                            va='center', ha='left', color=color, fontsize=6,
                            bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.2'))
                else:

                    plt.text(mean_val, vertical_offset, f'{type_} mean: {mean_val:.1%}',
                            va='center', ha='left', color=color, fontsize=6,
                            bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.2'))


        # Add title and labels
        plt.title('Distribution of Metrics', fontsize=16)
        plt.xlabel('Value', fontsize=14)
        plt.ylabel('Metrics', fontsize=14)
        plt.xlim(-0.05, 1.05)  
        if self.dump_files:
            plt.savefig(self.dump_path / 'overview_metrics.png')
        plt.show()
        
    def visualize_class_metrics_distribution_per_class(self, df):
        df_metrics_sample = pd.DataFrame(columns=['sample_id', 'relation', 'f1', 'precision', 'recall'])
        
        for i, row in df.iterrows():
            failure_modes = row['failure_modes']  # Assuming this is a dictionary, not a string that needs to be parsed
            for relation, data in failure_modes.items():
                tp = data['counts']['tp']
                fp = data['counts']['fp']
                fn = data['counts']['fn']

                if tp + fp + fn == 0:
                    continue  # Skip this one, it's all zeros
                
                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / (tp + fn) if tp + fn > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                
                new_row = pd.DataFrame({
                    'sample_id': [i],
                    'relation': [relation],
                    'f1': [f1],
                    'precision': [precision],
                    'recall': [recall]
                })
                df_metrics_sample = pd.concat([df_metrics_sample, new_row], ignore_index=True)

        if self.dump_files:
            df_metrics_sample.to_csv(self.dump_path / 'df_metrics_sample.csv')
        

        for metric in self.metrics:
            df_filtered = df_metrics_sample[['sample_id', 'relation', metric]].copy()
            df_filtered[metric] = pd.to_numeric(df_filtered[metric], errors='coerce')
            df_filtered[metric] = np.ma.filled(df_filtered[metric], np.nan)
            df_filtered.dropna(subset=[metric], inplace=True)
            if metric == 'f1':
                f1_df = df_filtered


        # Make sure the metrics are numeric and NaNs are handled
        df_metrics_sample[self.metrics] = df_metrics_sample[self.metrics].apply(pd.to_numeric, errors='coerce')
        df_metrics_sample.dropna(subset=self.metrics, inplace=True)

        # Melt the DataFrame
        df_metrics_sample_melted = df_metrics_sample.melt(id_vars=['sample_id', 'relation'], value_vars=self.metrics, var_name='metric', value_name='value')
        def custom_sort(relation):
            return (relation != 'null_relation', relation)

        df_metrics_sample_melted['sort_order'] = df_metrics_sample_melted['relation'].apply(custom_sort)
        df_metrics_sample_melted = df_metrics_sample_melted.sort_values('sort_order').drop('sort_order', axis=1)

        if self.dump_files:
            df_metrics_sample_melted.to_csv(self.dump_path / 'df_metrics_sample_melted.csv')
                                     
        # Create the plot
        _ = plt.figure(figsize=(8, 8/11*len(self.relations)))  # Adjust for a portrait layout

        # Violin plot
        sns.violinplot(y='relation', x='value', hue='metric', data=df_metrics_sample_melted, inner=None, palette="coolwarm", cut=0)

        # Dots with edge color for visibility
        sns.stripplot(y='relation', x='value', hue='metric', data=df_metrics_sample_melted, dodge=True, size=5, alpha=0.5, edgecolor="black", linewidth=0.5, palette="coolwarm")


        plt.title('Distribution of Metrics by Relation')
        plt.ylabel('Relation')  
        plt.xlabel('Metric Value')

        # Existing code to plot...
        legend = plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Assuming xlim is set or known
        xlim_max = plt.gca().get_xlim()[1]  

        mean_f1_dict = {}
        new_y_labels = []

        for rel in self.relations:
            metric = 'f1'
            if 'f1_df' in locals():
                mean_val = self.metrics_dict['per_class'].get(rel, {}).get(metric, None)
                mean_f1_dict[rel] = mean_val


        current_yticks, current_yticklabels = plt.yticks()

        # Make a list to hold new labels
        new_y_labels = []

        # Loop through the current y-tick labels and modify them
        for idx, label in enumerate(current_yticklabels):
            rel = label.get_text()  # Grabbing the 'relation' from the current label
            mean_val = mean_f1_dict.get(rel, None)  # Fetch the F1 mean value
            new_label = f"{rel}\nF1 Mean: {mean_val:.1%}" if mean_val is not None else f"{rel}\nF1 Mean: None"
            new_y_labels.append(new_label)

        # Update y-ticks and their labels
        plt.yticks(ticks=current_yticks, labels=new_y_labels)

        # Now plot the scatter plot based on the newly-ordered relations
        for idx, label in enumerate(current_yticklabels):
            rel = label.get_text().split('\n')[0]
            mean_val = mean_f1_dict.get(rel, None)
            if mean_val is None:
                warnings.warn(f"Mean value is None for {rel}. Skipping plotting for this value.")
                plt.scatter(0, idx, color='darkblue', s=100, zorder=3)  # Use idx as the y-coordinate
            else:
                plt.scatter(mean_val, idx, color='darkblue', s=100, zorder=3)  # Use idx as the y-coordinate


        # This will return existing legend items
        handles, labels = plt.gca().get_legend_handles_labels()

        # Filter out the ones related to violin plot, assuming stripplot labels are 'coolwarm_0', 'coolwarm_1', etc.
        new_handles = handles[:3]

        # Add the F1 scatterplot to the handles
        new_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', markersize=10))

        # Add new label for it
        new_labels = labels[:3] + ['F1 Mean']

        # Now set the new legend
        lgd = plt.legend(handles=new_handles, labels=new_labels, title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.xlim(-0.05, 1.05)   
        plt.tight_layout(rect=[0, 0, 0.8, 1])  
        
        if self.dump_files:
            plt.savefig(self.dump_path / 'per_class_metrics.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
                    
        
        plt.show()
        
        return df_metrics_sample


    def dump_metrics(self):
        
        # Export the results to a JSON file
        self.metrics_dict['exp_group'] = self.exp_group
        if self.dump_files:
            class_metrics_path = self.dump_path / 'class_metrics.json'
            with open(class_metrics_path, 'w') as f:
                json.dump(self.metrics_dict, f)

            print(f'File exported: {class_metrics_path}')

            # Check if data_readme exists
            if os.path.exists(self.data_readme):
                outpath = self.dump_path / os.path.basename(self.data_readme)
                # Copy it to the dump_path
                shutil.copy(self.data_readme, outpath)
                
            dump_path = self.dump_path / 'report.json'
            self.df.to_json(dump_path, orient='records', lines=True)
            
            print(f'File exported: {dump_path}')
            
            if 'clsTskOnl' in self.data_stem:
                fix_cls_metrics_dump(self.data_stem, self.model_name)

        return self.metrics_dict



def load_and_process_data(data_folder, data_cap=-1, memorization_task=False, merge_train_dev=False):
    set_data = None
    dataset_sets = {}
    dict_sets = {}

    for set_ in ('train', 'test', 'dev'):

        data_path = os.path.join(data_folder, f'{set_}.json')

        with open(data_path, 'r') as f:
            data = json.load(f)
                
        # Remap keys and separate into train/test
        if memorization_task:
            if not set_data:
                set_data = [{"text": item["input"], "summary": item["output"], "title": ""} for item in data[data_cap:]]
        else:
            set_data = [{"text": item["input"], "summary": item["output"], "title": ""} for item in data]
            
        # Merge 'train' and 'dev' if the flag is set
        if merge_train_dev:
            if set_ == 'dev':
                dict_sets['train'] = dict_sets['train'] + set_data
            else:
                dict_sets[set_] = set_data
        else:
            dict_sets[set_] = set_data
            
    for set_ in ('train', 'test', 'dev'):
        if merge_train_dev:
            if set_ == 'dev':
                continue
        set_data = dict_sets[set_]
        dataset_sets[set_] = Dataset.from_dict(
            {"text": [item["text"] for item in set_data],
             "summary": [item["summary"] for item in set_data],
             "title": [item["title"] for item in set_data]}
            )

    # Create DatasetDict
    dataset_dict = DatasetDict(dataset_sets)
    
    return dataset_dict


def parse_json_objects(string):
    objects = []
    brace_count = 0
    start_index = 0

    for i, char in enumerate(string):
        if char == '{':
            if brace_count == 0:
                start_index = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                try:
                    obj = json.loads(string[start_index:i+1])
                    objects.append(obj)
                except json.JSONDecodeError:
                    # Handle or log parsing error if needed
                    pass

    return objects
