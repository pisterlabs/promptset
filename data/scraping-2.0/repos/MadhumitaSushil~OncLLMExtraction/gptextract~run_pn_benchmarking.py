import argparse
import ast
import csv
import ipdb
import os
from collections import defaultdict
from datetime import date

import pandas as pd

from gptextract.dataprocessor.annots_info import AnnotsInfo
from gptextract.dataprocessor.annots_for_inference import InferenceAnnots
from gptextract.bratreader.brat import Collection
from gptextract.prompts.prompt_design import Prompt
from gptextract.modeling.openai_api_azure import OpenaiApiCall
from gptextract.modeling.hf_pipeline import HuggingfaceModel
from gptextract.utils.parse_output import parse_entity_output, parse_namedtuples
from gptextract.utils.metrics import Metrics

from gptextract import *


class Annotation:
    def __init__(self, doc_idx, section_name, section_text, inf_type, inf_subtype, annotator, annotations):
        self.doc_idx = doc_idx
        self.section_name = section_name
        self.section_text = section_text
        self.inf_type = inf_type
        self.inf_subtype = inf_subtype
        self.annotator_idx = annotator
        self.annotations = annotations

    def serialize(self, fannot, dir_annot):
        headers = ['doc_idx', 'section_name', 'section_text', 'inference_type', 'inference_subtype',
                   'annotator_idx', 'annotation_set']

        data = [self.doc_idx, self.section_name, self.section_text, self.inf_type, self.inf_subtype,
                self.annotator_idx, self.annotations]

        with open(os.path.join(dir_annot, fannot), 'a') as f:
            csvwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if f.tell() == 0:
                # write header to empty file
                csvwriter.writerow(headers)
            csvwriter.writerow(data)


class AnnotationColl:

    def __init__(self, annot_lst):
        self.annotation_coll = annot_lst

        self.processed = {cur_annot.doc_idx + cur_annot.section_name
                          + cur_annot.inf_type + cur_annot.inf_subtype
                          + cur_annot.annotator_idx
                          for cur_annot in self.annotation_coll}

        self.doc2annotator = defaultdict(set)
        for cur_annot in self.annotation_coll:
            self.doc2annotator[cur_annot.doc_idx].add(cur_annot.annotator_idx)

    @classmethod
    def read_existing_coll(cls, fname_coll, dir_coll):
        df = pd.read_csv(os.path.join(dir_coll, fname_coll), quotechar='"')
        df.drop_duplicates(inplace=True)
        annot_lst = list()

        for row in df.itertuples(index=False):
            if row.inference_type == 'entity':
                annots = ast.literal_eval(row.annotation_set.strip())
            elif row.inference_type == 'advanced_inference':
                annots = [eval(cur_annot) for cur_annot in row.annotation_set.strip().split('\n')]
            else:
                raise ValueError("Unsupported inference type")

            cur_annot = Annotation(row.doc_idx, row.section_name, row.section_text,
                                   row.inference_type, row.inference_subtype, row.annotator_idx, annots)

            annot_lst.append(cur_annot)

        return cls(annot_lst)

    def is_processed(self, example):
        if example in self.processed:
            return True
        else:
            return False

    def is_annotated(self, doc_idx, annotator_idx):
        """
        Check if doc_idx has been annotated by annotator_idx.
        :return: True if annotated, False otherwise
        """
        if doc_idx in self.doc2annotator and annotator_idx in self.doc2annotator[doc_idx]:
            return True
        return False

    def get_annot_from_coll(self, doc_idx, section_name, inf_type, inf_subtype, annotator_idx):
        matching_annots = list()
        for annot in self.annotation_coll:
            if (annot.doc_idx == doc_idx and annot.section_name == section_name and
                    annot.inf_type == inf_type and annot.inf_subtype == inf_subtype and
                    annot.annotator_idx == annotator_idx):
                matching_annots.append(annot)

        return matching_annots


class Output:
    def __init__(self, doc_idx, section_name, section_text, inf_type, inf_subtype,
                 model_type, temperature, prompt_preamble, prompt,
                 output):
        self.doc_idx = doc_idx
        self.section_name = section_name
        self.section_text = section_text
        self.inf_type = inf_type
        self.inf_subtype = inf_subtype

        self.model_type = model_type
        self.temp = temperature
        self.prompt_preamble = prompt_preamble
        self.prompt = prompt
        self.output = output

    def serialize(self, fname_output, dir_output):
        headers = ['doc_idx', 'section_name', 'section_text', 'inference_type', 'inference_subtype',
                   'model_type', 'temperature', 'prompt_preamble', 'prompt', 'model_output'
                   ]

        data = [self.doc_idx, self.section_name, self.section_text, self.inf_type, self.inf_subtype,
                self.model_type, self.temp, self.prompt_preamble, self.prompt, self.output]

        with open(os.path.join(dir_output, fname_output), 'a') as f:
            csvwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if f.tell() == 0:
                # write header to empty file
                csvwriter.writerow(headers)
            csvwriter.writerow(data)


class OutputColl:
    def __init__(self, output_lst):
        self.outputs = output_lst

        self.processed = {cur_out.doc_idx + cur_out.section_name + cur_out.inf_type + cur_out.inf_subtype +
                          str(cur_out.model_type) + str(cur_out.temp) + cur_out.prompt_preamble + cur_out.prompt
                          for cur_out in self.outputs}

    @classmethod
    def read_existing_coll(cls, fname_coll, dir_coll):
        df = pd.read_csv(os.path.join(dir_coll, fname_coll), quotechar='"')
        output_lst = list()

        for row in df.itertuples(index=False):
            if row.inference_type == 'entity':
                parsed_output = parse_entity_output(row.model_output)
            elif row.inference_type == 'advanced_inference':
                parsed_output = parse_namedtuples(row.model_output, row.inference_subtype)
            else:
                raise ValueError("Unsupported inference type")

            if str(row.prompt_preamble) == 'nan':
                prompt_preamble = ''
            else:
                prompt_preamble = row.prompt_preamble

            cur_out = Output(row.doc_idx, row.section_name, row.section_text,
                             row.inference_type, row.inference_subtype, row.model_type, row.temperature,
                             prompt_preamble, row.prompt, parsed_output
                             )

            output_lst.append(cur_out)

        return cls(output_lst)

    def is_processed(self, example):
        if example in self.processed:
            return True
        else:
            return False


class OncInfoExtr:
    def __init__(self, sections_to_infer=('hpi', 'a&p'),
                 annotators=('expert', 'student1'),
                 inference_types=('entity', 'advanced_inference',),
                 fdata='onc_pn_ie_data.csv', foutput='onc_pn_ie_output.csv', fscores='scores.csv',
                 data_dir='../../data/', output_dir='../../output/',
                 append_timestamp=True):

        annots_info = AnnotsInfo()

        self.ie_texts = dict()  # {section_name: {annotated_fname: section_text}}
        for key, values in annots_info.annotated_section_text.items():
            if key in sections_to_infer:
                self.ie_texts[key] = values

        self.inference_types = inference_types
        self.annotators = annotators

        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(data_dir)

        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(output_dir)

        self.fname_ie_data = fdata
        if append_timestamp:
            self.fname_ie_data = self.fname_ie_data.rstrip('csv').rstrip('.') + '_' + str(date.today()) + '.csv'

        self.fname_ie_output = foutput
        if append_timestamp:
            self.fname_ie_output = self.fname_ie_output.rstrip('csv').rstrip('.') + '_' + str(date.today()) + '.csv'

        self.fname_scores = fscores
        if append_timestamp:
            self.fname_scores = self.fname_scores.rstrip('csv').rstrip('.') + '_' + str(date.today()) + '.csv'

    def get_elements_for_inference(self, entities, relations, attributes):
        inference_annots = InferenceAnnots(entities, relations, attributes)

        entity_mentions = inference_annots.get_entity_mentions_for_inference()
        advanced_inference_tuple_dict = inference_annots.get_tuples_for_advanced_inference()

        return entity_mentions, advanced_inference_tuple_dict

    def create_ie_dataset(self, coll_dir, annotator_idx):
        collection = Collection.read_collection(coll_dir)

        if os.path.exists(os.path.join(self.data_dir, self.fname_ie_data)):
            existing_coll = AnnotationColl.read_existing_coll(self.fname_ie_data, self.data_dir)
        else:
            existing_coll = None
        
        for section, texts in self.ie_texts.items():
            for doc_idx, cur_sec_text in texts.items():
                cur_sec_text = cur_sec_text.replace('', ' ')
                document = collection.get_document(doc_idx)
                if document is None:
                    print("Document ", doc_idx, " not annotated by ", annotator_idx)
                    continue
                doc_text = document.text
                section_start = doc_text.index(cur_sec_text)
                section_end = section_start + len(cur_sec_text)

                entities, relations, atts = collection.get_section_annots(doc_idx, section_start, section_end)

                inference_entities, adv_inf_tuples = self.get_elements_for_inference(entities, relations, atts)

                for cur_ent_type, annotated_entities in inference_entities.items():
                    if existing_coll and existing_coll.is_processed(
                            doc_idx + section + 'entity' + cur_ent_type + annotator_idx):
                        print("Skipping duplicate data entry")
                        continue

                    # serialize the input data
                    annot_obj = Annotation(doc_idx, section, cur_sec_text, 'entity', cur_ent_type,
                                        annotator_idx, annotated_entities)
                    annot_obj.serialize(self.fname_ie_data, self.data_dir)

                for cur_inf_type, tuple in adv_inf_tuples.items():
                    str_tuple = ""
                    for cur_item in tuple:
                        str_tuple += str(cur_item)+'\n'

                    if existing_coll and existing_coll.is_processed(
                            doc_idx + section + 'advanced_inference' + cur_inf_type + annotator_idx):
                        print("Skipping duplicate data entry")
                        continue

                    annot_obj = Annotation(doc_idx, section, cur_sec_text, 'advanced_inference', cur_inf_type,
                                           annotator_idx, str_tuple)
                    # serialize the input data
                    annot_obj.serialize(self.fname_ie_data, self.data_dir)

    def extract_information(self, model='gpt-35-turbo', backend='openai-azure'):
        if backend == 'openai-azure':
            model_obj = OpenaiApiCall(model=model, temperature=0., max_tokens=512)
        else:
            model_obj = HuggingfaceModel(model_name_or_path=model)

        if os.path.exists(os.path.join(self.output_dir, self.fname_ie_output)):
            existing_coll = OutputColl.read_existing_coll(self.fname_ie_output, self.output_dir)
        else:
            existing_coll = None

        prompt_obj = Prompt()

        for cur_inference_type in self.inference_types:
            prompt_preamble = prompt_obj.get_prompt_preamble(model_obj.model_name, cur_inference_type)
            for cur_inference_subtype, prompt in prompt_obj.yield_inference_subtype_prompt(cur_inference_type):
                for cur_section_type, cur_texts_dict in self.ie_texts.items():
                    for cur_doc_idx, cur_sec_text in cur_texts_dict.items():
                        cur_example = cur_doc_idx + cur_section_type + cur_inference_type + \
                                      cur_inference_subtype + model_obj.model_name + str(model_obj.temperature) + \
                                      prompt_preamble + prompt
                        if existing_coll and existing_coll.is_processed(cur_example):
                            print("Instance already processed; continuing.")
                            continue
                        cur_prompt = cur_sec_text + prompt

                        # ipdb.set_trace()
                        # get model output
                        result = model_obj.get_response_text(cur_prompt, prompt_preamble)
                        # try:
                        #     result = model_obj.get_response_text(cur_prompt, prompt_preamble)
                        # except Exception as e:
                        #     print(e)
                        #     break

                        print("Response: ", result)
                        output_obj = Output(cur_doc_idx, cur_section_type, cur_sec_text,
                                            cur_inference_type, cur_inference_subtype,
                                            model_obj.model_name, model_obj.temperature, prompt_preamble,
                                            prompt, result)
                        # save the result
                        output_obj.serialize(self.fname_ie_output, self.output_dir)


    @staticmethod
    def _format_entity_annot_for_eval(annotation, inference_subtype):

        annotation = [item.lower() for item in annotation]

        if inference_subtype.lower() in ['metastasis', 'remission', 'hospice', 'radiationtherapyname'] \
                and len(annotation):
            annotation = ['yes']

        if not len(annotation):
            annotation = ['none', 'no', 'n/a', 'unknown']

        return annotation

    @staticmethod
    def _format_tuple_annots_for_eval(tuple_list):
        relations = defaultdict(set)

        for cur_tuple in tuple_list:
            if type(cur_tuple) == CancerDiagnosis:
                # print("skipping first cancer diagnosis")
                continue

            tuple_primary_type = cur_tuple._fields[0]
            tuple_primary_val = cur_tuple[0]

            for i, (name, vals) in enumerate(cur_tuple._asdict().items()):
                if not i or name == 'AdditionalTesting':
                    # print("Skipping additional testing")
                    continue
                if len(vals) == 0:
                    vals = 'unknown'
                if type(vals) == set:
                    for val in vals:
                        relations[tuple_primary_type + ' ' + name].add(tuple_primary_val + ' ' + val)
                else:
                    relations[tuple_primary_type + ' ' + name].add(tuple_primary_val + ' ' + vals)

        if not len(relations):
            raise ValueError("Empty output set", tuple_list)

        return relations

    def evaluate(self):
        metrics = Metrics(tokenizer='default')
        scores = list()

        annots_coll = AnnotationColl.read_existing_coll(self.fname_ie_data, self.data_dir)
        output_coll = OutputColl.read_existing_coll(self.fname_ie_output, self.output_dir)

        for cur_output in output_coll.outputs:
            for annotator_idx in self.annotators:
                if not annots_coll.is_annotated(cur_output.doc_idx, annotator_idx):
                    continue
                matching_annots = annots_coll.get_annot_from_coll(cur_output.doc_idx, cur_output.section_name,
                                                                  cur_output.inf_type, cur_output.inf_subtype,
                                                                  annotator_idx)

                if not len(matching_annots):  # the annotation does not exist in the document
                    if cur_output.inf_type == 'entity':
                        matching_annots = [
                            Annotation(
                                doc_idx=cur_output.doc_idx,
                                section_name=cur_output.section_name,
                                section_text=cur_output.section_text,
                                inf_type=cur_output.inf_type,
                                inf_subtype=cur_output.inf_subtype,
                                annotator=annotator_idx,
                                annotations={'none', 'no', 'n/a', 'unknown'}
                            )
                        ]
                    elif cur_output.inf_type == 'advanced_inference':
                        matching_annots = [
                            Annotation(
                                doc_idx=cur_output.doc_idx,
                                section_name=cur_output.section_name,
                                section_text=cur_output.section_text,
                                inf_type=cur_output.inf_type,
                                inf_subtype=cur_output.inf_subtype,
                                annotator=annotator_idx,
                                annotations=[inference_subtype_to_default_ne_dict[cur_output.inf_subtype]]
                            )
                        ]

                if len(matching_annots) > 1:
                    raise ValueError("More than one annotation for this case", matching_annots)

                annot = matching_annots[0]

                # for annot in matching_annots:
                annotations = annot.annotations

                if annot.inf_type == 'entity':
                    annotations = self._format_entity_annot_for_eval(annotations, annot.inf_subtype)
                    outputs = [item.lower() for item in cur_output.output]
                    if not len(outputs):
                        outputs = ['none']

                    annots_for_scoring = [annotations for _ in range(len(outputs))]

                    bleu = metrics.compute_bleu_score(outputs, annots_for_scoring)['bleu']

                    rouge1 = metrics.compute_rouge_score(outputs, annots_for_scoring,
                                                         rouge_types=['rouge1'])['rouge1']

                    cur_scores = {
                        'doc_idx': annot.doc_idx,
                        'section_name': annot.section_name,
                        'inference_type': annot.inf_type,
                        'inference_subtype': annot.inf_subtype,
                        'model_type': cur_output.model_type,
                        'temp': cur_output.temp,
                        'prompt': cur_output.prompt,
                        'cur_pred_type': 'Unknown',
                        'annotator_idx': annotator_idx,
                        'bleu': bleu,
                        'rouge1': rouge1,

                    }
                    scores.append(cur_scores)

                elif annot.inf_type == 'advanced_inference':
                    annotations = self._format_tuple_annots_for_eval(annotations)
                    outputs = self._format_tuple_annots_for_eval(cur_output.output)

                    for cur_pred_type, cur_pred_vals in outputs.items():
                        cur_pred_vals = [item.lower() for item in cur_pred_vals]
                        cur_annot_vals = annotations[cur_pred_type]
                        if not len(cur_annot_vals):
                            # may never be used; retaining regardless.
                            cur_annot_vals = ['none', 'no', 'n/a', 'unknown']
                        else:
                            cur_annot_vals = [item.lower() for item in cur_annot_vals]
                        annots_for_scoring = [cur_annot_vals for _ in range(len(cur_pred_vals))]

                        bleu = metrics.compute_bleu_score(cur_pred_vals, annots_for_scoring)['bleu']

                        rouge1 = metrics.compute_rouge_score(cur_pred_vals, annots_for_scoring,
                                                             rouge_types=['rouge1'])['rouge1']

                        cur_scores = {
                            'doc_idx': annot.doc_idx,
                            'section_name': annot.section_name,
                            'inference_type': annot.inf_type,
                            'inference_subtype': annot.inf_subtype,
                            'model_type': cur_output.model_type,
                            'temp': cur_output.temp,
                            'prompt': cur_output.prompt,
                            'cur_pred_type': cur_pred_type,
                            'annotator_idx': annotator_idx,
                            'bleu': bleu,
                            'rouge1': rouge1,

                        }
                        scores.append(cur_scores)

        scores = pd.DataFrame(scores)
        scores.to_csv(os.path.join(self.output_dir, self.fname_scores), index=False)
        self.aggregate_scores()

        return scores

    def aggregate_scores(self):
        df = pd.read_csv(os.path.join(self.output_dir, self.fname_scores))

        df = df.groupby(['inference_type', 'inference_subtype', 'cur_pred_type',
                         'annotator_idx', 'model_type', 'temp', 'prompt',
                         ]).agg({'bleu': ['mean'],
                                 'rouge1': ['mean']}
                                )

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(df)

    def quantify_common_annotated_reports(self):
        common_df = list()

        df = pd.read_csv(os.path.join(self.output_dir, self.fname_scores))

        for cur_idx in df['doc_idx'].unique():
            cur_df = df[df['doc_idx'] == cur_idx]

            cur_annotators = cur_df['annotator_idx'].unique()

            if len(cur_annotators) == 2:
                common_df.append(cur_df)

        df = pd.concat(common_df)

        df = df.groupby(['inference_type', 'inference_subtype', 'cur_pred_type',
                         'annotator_idx', 'model_type', 'temp', 'prompt',
                         ]).agg({'bleu': ['mean'],
                                 'rouge1': ['mean']}
                                )

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print("Results on common annotated reports only: ")
        print(df)

    def remove_useless_outputs(self):
        df_scores = pd.read_csv(os.path.join(self.output_dir, self.fname_scores))
        prompt_obj = Prompt()
        new_df = list()

        for cur_type in df_scores['inference_subtype'].unique():
            cur_df = df_scores[df_scores['inference_subtype'] == cur_type]
            for cur_prompt in cur_df['prompt'].unique():
                if str(cur_prompt) == 'nan':
                    continue
                if len(cur_df['inference_type'].unique()) > 1:
                    raise ValueError("More than one inference type!", cur_df)
                if (cur_type in prompt_obj.adv_criteria_to_prompt_mapping or
                    cur_type in prompt_obj.entity_to_prompt_mapping) and (
                        cur_prompt == prompt_obj.get_prompt(cur_df['inference_type'].unique()[0], cur_type)):
                    new_df.append(cur_df[cur_df['prompt'] == cur_prompt])
                else:
                    print(cur_prompt)

        new_df = pd.concat(new_df)
        new_df.to_csv(os.path.join(self.output_dir, self.fname_scores), index=False)


def main(coll_dir, model='gpt-35-turbo', do_eval=True):
    ie_extractor = OncInfoExtr(
        inference_types=('advanced_inference',),  # important to add , in the end to avoid character iteration
        annotators=coll_dir.keys(),
        fdata='onc_pn_ie_data_2023-03-31.csv',
        foutput='onc_pn_ie_output_2023-03-31.csv',
        fscores='scores_2023-03-31.csv',
        append_timestamp=False)

    for annotator_idx in coll_dir.keys():
        ie_extractor.create_ie_dataset(coll_dir[annotator_idx], annotator_idx)

    if 'gpt-35' in model or 'gpt-4' in model:
        ie_extractor.extract_information(model=model, backend='openai-azure')
    else:
        ie_extractor.extract_information(model=model, backend='hf')

    if do_eval:
        ie_extractor.evaluate()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-annotations_path",
                        default='../data/all_annotated/',
                        type=str,
                        required=False,
                        help="Annotated data path")

    parser.add_argument("-annotator",
                        default='all_annots',
                        type=str,
                        required=False,
                        help="Annotator name")

    parser.add_argument("-model",
                        default='gpt-35-turbo',
                        type=str,
                        required=True,
                        help="Model name")

    parser.add_argument("--evaluate", action="store_true", help="Evaluates the model")

    args = parser.parse_args()

    coll_dir = {
       args.annotator: args.annotations_path,
    }

    main(coll_dir, args.model, args.evaluate)

