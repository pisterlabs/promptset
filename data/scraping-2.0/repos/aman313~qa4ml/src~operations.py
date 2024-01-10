import csv
import random
from typing import List, Iterable, Union, Dict, Callable
import openai

from src.constants import openai_key, openai_engine

openai.api_key = openai_key

from src.models import Prompt, FoundationModelAPIWrapper, PromptBasedCaseDetectionCriterion, DataPoint, \
    TextClassificationDataPoint, Subset


class Operation():

    def validate_inputs(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Sampler(Operation):
    pass

class RandomSampler(Sampler):

    def __init__(self, sampling_ratio:Union[int,float],seed:int=42):
        self._sampling_ratio = sampling_ratio
        self.sampler_seeded = random.Random(seed)

    def __call__(self, iter:Iterable ):
        sampled = []
        for elem in iter:
            sample = self.sampler_seeded.random()
            if sample<self._sampling_ratio:
                sampled.append(elem)
        return sampled

class ConditionSampler(Sampler):

    def __init__(self, matching_function:Callable, max_samples = -1):
        self._matching_function = matching_function
        self._max_samples =  -1

    def __call__(self, iter:Iterable):
        sampled  = []
        for elem in iter:
            if self._matching_function(elem) and (self._max_samples ==-1 or (self._max_samples>-1 and len(sampled) < self._max_samples)):
                sampled.append(elem)
        return sampled


class HTTPOperation(Operation):

    def _get(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

class CaseDetectionCriteriaCreator(Operation):
    pass

class PromptBasedCaseDetectionCriteriaCreator(CaseDetectionCriteriaCreator):

    def __init__(self, prompt:Prompt):
        self._prompt = prompt

    def __call__(self)->PromptBasedCaseDetectionCriterion:
        return PromptBasedCaseDetectionCriterion(self._prompt)


class CriterionApplier(Operation):
    pass

class FoundationModelCriterionApplier(CriterionApplier):

    def __init__(self, api_wrapper:FoundationModelAPIWrapper, criterion:PromptBasedCaseDetectionCriterion):
        self._api_wrapper = api_wrapper
        self._criterion = criterion

    def __call__(self, data_points:List[TextClassificationDataPoint]):
        for data_point in data_points:
            augmented_prompt_text = self._criterion._prompt.augment_text(data_point)
            engine = openai_engine
            kwargs = self._api_wrapper._api_details
            response = openai.Completion.create(engine=engine,
                                                prompt=augmented_prompt_text,
                                                **kwargs
                                                )
            yield response.choices[0].text



class SubsetCreator(Operation):
    pass

class SubsetCreatorFromCriteriaCreator(SubsetCreator):
    def __init__(self, criteria_creator:CaseDetectionCriteriaCreator):
        self._criteria_creator = criteria_creator

class PromptBasedSubsetCreatorFromCriteriaCreator(SubsetCreatorFromCriteriaCreator):

    def __init__(self, prompt_criteria_creator:PromptBasedCaseDetectionCriteriaCreator,
                 foundation_model_api_wrapper:FoundationModelAPIWrapper):
        super().__init__(prompt_criteria_creator)
        self._criterion = prompt_criteria_creator()
        self._foundation_model_api_wrapper = foundation_model_api_wrapper

    def __call__(self, data_points:List[DataPoint]):
        applier = FoundationModelCriterionApplier(self._foundation_model_api_wrapper,self._criterion)
        prompted_classes = list(applier(data_points))
        prompted_class_dict = {}
        for i in range(len(prompted_classes)):
            try:
                prompted_class_dict[prompted_classes[i]].append(data_points[i])
            except KeyError:
                prompted_class_dict[prompted_classes[i]] = [data_points[i]]
        subsets = []
        for cls,data in prompted_class_dict.items():
            subsets.append(Subset(cls,data))
        return subsets

class Summarizer(Operation):
    pass

class SubsetCountSummarizer(Summarizer):
    def __init__(self,verbose=False):
        self._verbose=verbose

    def __call__(self, subsets:List[Subset]):
        if self._verbose:
            for x in subsets:
                print('\n')
                print(x._name)
                for i in range(len(x._data_points)):
                    d = x._data_points[i]
                    print(str(i) + ': ' + str(d))
        return {x._name:len(x._data_points) for x in subsets}


class DatasetReader(Operation):
    def __call__(self, dataset_location:str)->Iterable[DataPoint]:
        raise NotImplementedError

class TextClassificationDatasetReader(DatasetReader):

    def __call__(self, dataset_location:str)->Iterable[TextClassificationDataPoint]:
        raise NotImplementedError

class ToxicCommentDatasetReader(TextClassificationDatasetReader):

    def __call__(self, dataset_location:str)->Iterable[TextClassificationDataPoint]:
        raise NotImplementedError


class DisasterTweetDatasetReader(TextClassificationDatasetReader):
    def __call__(self, dataset_location):
        with open(dataset_location) as dl:
            reader = csv.DictReader(dl)
            for row in reader:
                text = row['text']
                label = row['target']
                metadata = {'location':row['location'],'keyword':row['keyword']}

                yield TextClassificationDataPoint(text=text,label=label,metadata=metadata)