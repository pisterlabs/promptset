import dataclasses
from dataclasses import field

#            -= OpenAI text data =-
# {
#  "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
#  "object": "text_completion",
#  "created": 1589478378,
#  "model": "text-babbage-001",
#  "choices": [
#    {
#      "text": "\n\nThis is a test",
#      "index": 0,
#      "logprobs": null,
#      "finish_reason": "length"
#    }
#  ],
#  "usage": {
#    "prompt_tokens": 5,
#    "completion_tokens": 6,
#    "total_tokens": 11
#  }
# }

#            -= OpenAI embeddings response =-
# {
#  "object": "list",
#  "data": [
#    {
#      "object": "embedding",
#      "embedding": [
#        0.018990106880664825,
#        -0.0073809814639389515,
#        .... (1024 floats total for ada)
#        0.021276434883475304,
#      ],
#      "index": 0
#    }
#  ],
#  "usage": {
#    "prompt_tokens": 8,
#    "total_tokens": 8
#  }
# }

# this class organizes the data received from the llm api in a manner that the consumer of the wrapper can rely on.
from typing import Any


@dataclasses.dataclass
class LLMResponse:
    """
    """
    raw_response: 'typing.Any' = object()
    text: list = field(default_factory=list)
    text_processed_data: dict = field(default_factory=dict)
    data: list = field(default_factory=list)
    data_processed: list = field(default_factory=list)


@dataclasses.dataclass
class LLMDefaults:
    default_completion_model_name: str = None
    default_search_query_model_name: str = None
    default_search_document_model_name: str = None


class OpenaiLLMDefaults():
    default_completion_model_name: str = "text-ada-001"
    default_search_query_model_name: str = "text-search-babbage-query-001"
    default_search_document_model_name: str = "babbage-search-document"


@dataclasses.dataclass
class LLMRequest:
    """
    main data interface to the LLMWrapper class. stores data that is sent to the LLM and
    results from filters and pre- / post-processing.
    """
    temperature: float = .5
    max_tokens: int = 40
    top_p: float = .7
    best_of: int = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0
    stop: str = "."
    prompt: list = field(default_factory=list)
    query: list = field(default_factory=list)
    context: list = field(default_factory=list)
    documents: list = field(default_factory=list)

    prompt_processed_data: dict = field(default_factory=dict)
    query_processed_data: dict = field(default_factory=dict)
    context_processed_data: dict = field(default_factory=dict)
    documents_processed_data: dict = field(default_factory=dict)

    n: int = 1

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ["prompt", "query", "context", "documents"]:
            if value.__class__ == str:
                super().__setattr__(name, [value])
            elif value.__class__ == list:
                super().__setattr__(name, value)
            else:
                raise TypeError("LLMRequest.__setattr__() only accepts str or list as value")
        else:
            super().__setattr__(name, value)


@dataclasses.dataclass
class OpenaiKWArgs(LLMRequest):
    """KWArgs suitable for OPENAI"""
    temperature: float = .5
    max_tokens: int = 40
    top_p: float = .7
    best_of: int = 1
    frequency_penalty: float = .0
    presence_penalty: float = 0
    stop: str = "."
    prompt: str = None
    n: int = 1


class BaseLLMProcessor:
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __init__(self, name: str):
        self.name = name
    def get_reference_list(self,size):
        # returns a list of lists so as to enable passing of strings by reference
        return [[None] for i in range(size)]

class LLMReqProcessor(BaseLLMProcessor):
    def will_handle(*args):
        if issubclass(args[1], LLMRequest):
            return True
        return False

    def apply(self, request: LLMRequest, response: LLMResponse):
        raise NotImplementedError("apply() not implemented")

    def __call__(self, request: LLMRequest):

        if not request:
            raise RuntimeError("nothing to apply")
        else:

            if request.query and request.documents:  # search request
                data1 = [request.query]
                data2 = [[i] for i in request.documents]
                modify_list = data1 + data2

                report1 = []
                report2 = []

                request.query_processed_data[self.name] = report1
                request.docs_processed_data[self.name] = report2
                report_list = [report1] + [report2]

            if request.prompt:
                modify_list += [[i] for i in request.prompt]
                report_list += self.get_reference_list(len(request.prompt))
                request.prompt_processed_data[self.name] = report_list

            if request.context:
                modify_list += [[i] for i in request.context]
                report_list += self.get_reference_list(len(request.context))
                request.context_processed_data[self.name] = report_list

            self.apply(modify_list, report_list)


class LLMResProcessor(BaseLLMProcessor):
    def will_handle(*args):
        if issubclass(args[1], LLMResponse):
            return True
        return False
    def apply(self, request: LLMRequest, response: LLMResponse):
        raise NotImplementedError("apply() not implemented")

    def __call__(self, response: LLMResponse):
        report_list = []
        modify_list = []

        if not response:
            raise RuntimeError("nothing to apply")
        else:
            if response.text:
                modify_list += [[i.text] for i in response.choices]
                report_list += self.get_reference_list(len(response.text))

            if response.data:
                modify_list += [i.embedding for i in response.data]
                report_list += self.get_reference_list(len(response.data))

        self.apply(modify_list, report_list)


class LLMReqResProcessor(BaseLLMProcessor):
    """Superclass that all LLM filters should inherit from, subclasses should implement
    processor_func_single and processor_func_double methods
    """
    def will_handle(*args):
        if len(args) !=3: return False
        if issubclass( type(args[1]), LLMRequest):
            if issubclass(type(args[2]), LLMResponse):
                return True
        return False
    def apply(self, request: LLMRequest, response: LLMResponse):
        raise NotImplementedError("apply() not implemented")

    def __call__(self, request: LLMRequest, response: LLMResponse):
        # package the individual texts to be processed into lists, of lists to pass them around as objects
        # apply the processor_func
        # un-package and assign the reqeust or responses values to the modified data.
        if (not request) and (not response):
            raise RuntimeError("nothing to apply")
        # set the params
        else:
            modify_list1 = []
            report_list1 = []
            modify_list2 = []
            report_list2 = []

            if request:
                if request.prompt:  # prompt req/resp: for moderation, similarity, length...
                    # for moderation the texts can just be modified directly
                    # for length the texts dont need altered just scored in the reports_list
                    # for similarity the prompt needs to be compared to the text and we dont know how this instance
                    # of this filter will be used, so... we assume the subclass will implement some way to figure it
                    # out for their use case

                    modify_list1 = [[i] for i in request.prompt]
                    report_list1 = [[None] for i in range(len(request.prompt))]

                    request.prompt = modify_list1
                    request.prompt_processed_data[self.name] = report_list1

            if response:
                if response.text:
                    modify_list2 = [[i] for i in response.text]
                    report_list2 = self.get_reference_list(len(response.text))

                    response.text = modify_list2
                    response.text_processed_data[self.name] = report_list2

            ret_val = self.apply(modify_list1+modify_list2, report_list1+report_list2)

        # now bring the dimensionality back one level
            if request:
                if request.prompt:
                    request.prompt = [i[0] for i in request.prompt]
                    request.prompt_processed_data[self.name] = [i[0] for i in request.prompt_processed_data[self.name]]
            if response:
                if response.text:
                    response.text = [i[0] for i in response.text]
                    response.text_processed_data[self.name] = [i[0] for i in response.text_processed_data[self.name]]
        return ret_val
        # types of use cases
        # . search request
        # . search response
        # . search pair
        # . completion request
        # . completion response
        # . completion pair
        # . embedding request
        # . embedding response
        # . embedding pair
        # . moderation request
        # . moderation response
        # . moderation pair
        # . error


class LLMWrapper:
    def __init__(self, api_name, api_key=None, completion_model_name=None, search_query_model_name=None,
                 search_document_model_name=None, completion_test_generator=None):
        """
        :param api_name: openai, or another provider name (only openai in this version)
        :param api_key: provide or leave blank for env variable
        """

        import os
        import openai
        self.completion_model_name = ""
        self.search_model_name = ""
        self.is_openai_api = False

        if api_name.lower() == "openai":
            self.is_openai_api = True
            # set default values for openai api
            self.set_defaults()
            # get the api key from the environment variable if it is not provided
            if not api_key:
                openai.api_key = os.getenv("OPENAI_API_KEY")
            else:
                openai.api_key = api_key
            # get the list of models
            self.models = openai.Model.list()["data"]
            self.API_KEY = openai.api_key
            self.authenticated = True

            if completion_model_name: self.completion_model_name = completion_model_name
            if search_query_model_name: self.search_query_model_name = search_query_model_name
            if search_document_model_name: self.search_document_model_name = search_document_model_name

        elif completion_test_generator:
            self.is_test_api = True
            self.completion_test_generator = completion_test_generator
            self.res_test_func = completion_test_generator

        else:
            raise Exception("Invalid API name")

    def set_defaults(self):
        # set the default values for the openai api
        # TODO: sure there is a programmatic way to do this
        if self.is_openai_api:
            if not self.completion_model_name:
                self.completion_model_name = OpenaiLLMDefaults.default_completion_model_name
            if not self.search_model_name:
                self.search_query_model_name = OpenaiLLMDefaults.default_search_query_model_name
                self.search_document_model_name = OpenaiLLMDefaults.default_search_document_model_name

    def handle_kwargs(self, request: LLMRequest) -> dict:
        """
        returns req modified to be compatible with the current api
        :rtype: dict
        """
        incoming_class = request.__class__
        if not incoming_class == LLMRequest:
            raise Exception("incoming class is not LLMRequest")

        if self.is_openai_api:
            assert (request.query.__class__ ==
                    request.prompt.__class__ ==
                    request.documents.__class__ ==
                    request.context.__class__ == list)
            oai_kwargs = {}

            if request.top_p is not None:
                oai_kwargs["top_p"] = request.top_p
            else:
                oai_kwargs["temperature"] = request.temperature

            oai_kwargs["max_tokens"] = request.max_tokens
            oai_kwargs["best_of"] = request.best_of
            oai_kwargs["frequency_penalty"] = request.frequency_penalty
            oai_kwargs["presence_penalty"] = request.presence_penalty
            oai_kwargs["stop"] = request.stop
            oai_kwargs["n"] = request.n

            oai_kwargs["query"] = request.query
            oai_kwargs["documents"] = request.documents

            if request.context:
                if not (len(request.context) == len(request.prompt)):
                    raise Exception("context and prompt arrays must be the same length")
                oai_kwargs["prompt"] = [request.context[i]+request.prompt[i] for i in range(len(request.context))]
            else:
                oai_kwargs["prompt"] = request.prompt

            return oai_kwargs

    def open_ai_search(self, request: LLMRequest) -> LLMResponse:
        if self.is_openai_api:
            import openai
            import numpy as np
            query_str = request["query"]

            if type(query_str) == list:
                if len(query_str) == 1:
                    query_str = query_str[0]
                else:
                    raise Exception("query must be a single string")

            query_embedding = openai.Embedding.create(input=query_str,
                                                      model=self.search_query_model_name).data[0].embedding
            choices_embeddings = openai.Embedding.create(input=request["documents"],
                                                         model=self.search_document_model_name).data
            choice_emb_tup = [
                (choice, choice_emb.embedding) for choice, choice_emb in zip(request["documents"], choices_embeddings)]

            def cos_sim(a, b):
                a = np.array(a)
                b = np.array(b)
                return (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

            lst_tup_sim_doc = [(cos_sim(query_embedding, choice_emb), choice) for choice, choice_emb in choice_emb_tup]
            lst_tup_sim_doc = sorted(lst_tup_sim_doc, key=lambda x: x[0], reverse=True)
            out = LLMResponse()
            out.text_processed_data["search"]=[]
            for r in lst_tup_sim_doc:
                out.text.append(r[1])
                out.text_processed_data["search"].append((r[0],r[1],query_str))
            return out

    def search(self, request: LLMRequest) -> LLMResponse:
        """
        returns the text response from the llm api
        :param request:
        :param prompt:
        :param req:
        :return:
        """

        if self.is_openai_api:
            import openai

            kwargs = self.handle_kwargs(request)

            if not issubclass(request.__class__, LLMRequest):
                raise Exception("Searches only possible with LLMRequest")
            else:
                result = self.open_ai_search(kwargs)

                return result

        elif self.is_other:
            raise Exception("not implemented")

    def completion(self, prompt=None, req: LLMRequest = None) -> LLMResponse:
        """
        returns the text response from the llm api, used for multiple completions
        :param prompt:
        :param req:
        :return: array of string completions
        """
        req = self.kwargs_check(req, prompt)

        if self.is_openai_api:
            if not issubclass(req.__class__, LLMRequest):
                raise Exception("keyword args class not for use with openai api")
            import openai

            kwargs_dict = self.handle_kwargs(req)

            kwargs_dict.pop("documents")
            kwargs_dict.pop("query")

            result = openai.Completion.create(model=self.completion_model_name,
                                              **kwargs_dict)

            out_result = LLMResponse(raw_response=result,
                                     text=[c.text for c in result["choices"]])

            return out_result
        elif self.is_test_api:
            return next(self.completion_test_generator)

        elif self.is_other:
            raise Exception("not implemented")

    def moderation(self, request: LLMRequest) -> LLMResponse:
        """
        returns the moderation response from the llm api
        :param request:
        :return:
        """
        if self.is_openai_api:
            import openai

            if not issubclass(request.__class__, LLMRequest):
                raise Exception("Moderation only possible with LLMRequest")
            else:
                result = openai.Moderation.create(input=request.query,
                                                  model=self.search_query_model_name)
                out_result = LLMResponse(raw_response=result,
                                         moderation=result["moderation"])

                return out_result

        elif self.is_other:
            raise Exception("not implemented")

    def kwargs_check(self, kwargs, prompt):
        if not prompt and not kwargs:
            raise Exception("No req provided")

        if kwargs:
            if issubclass(kwargs.__class__, LLMRequest):
                if (prompt is not None) and (kwargs.prompt is not None):
                    raise Exception("Prompt already provided")
                elif prompt is not None:
                    kwargs.prompt = prompt
                    prompt = None
                elif kwargs.prompt is not None:
                    # prompt already set correctly
                    pass
        else:
            kwargs = LLMRequest(prompt=prompt)
            prompt = None

        # check for compatible req
        return kwargs
