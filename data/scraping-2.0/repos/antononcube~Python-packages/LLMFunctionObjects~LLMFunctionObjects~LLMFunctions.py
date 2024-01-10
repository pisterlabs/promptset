import re
import warnings
import os
from LLMFunctionObjects.Configuration import Configuration
from LLMFunctionObjects.Evaluator import Evaluator
from LLMFunctionObjects.EvaluatorChat import EvaluatorChat
from LLMFunctionObjects.EvaluatorChatGPT import EvaluatorChatGPT
from LLMFunctionObjects.EvaluatorChatPaLM import EvaluatorChatPaLM
from LLMFunctionObjects.Functor import Functor
from LLMFunctionObjects.Chat import Chat
import openai
import google.generativeai
import warnings


# ===========================================================
# Configuration creation
# ===========================================================
def llm_configuration(spec, **kwargs):
    if spec is None:
        return llm_configuration('openai', **kwargs)
    elif isinstance(spec, Configuration):
        return spec.combine(kwargs)
    elif isinstance(spec, Evaluator):
        return llm_configuration(spec.conf, **kwargs)
    elif isinstance(spec, str) and spec.lower() == 'OpenAI'.lower():
        confOpenAI = Configuration(
            name="openai",
            api_key=None,
            api_user_id='user',
            module='openai',
            model='gpt-3.5-turbo-instruct',  # was 'text-davinci-003'
            function=openai.completions.create,  # was openai.Completion.create
            temperature=0.2,
            max_tokens=300,
            total_probability_cutoff=0.03,
            prompts=None,
            prompt_delimiter=' ',
            stop_tokens=None,
            argument_renames={"stop_tokens": "stop"},
            fmt='values',
            known_params=["api_key", "model", "prompt", "suffix", "max_tokens", "temperature", "top_p", "n", "stream",
                          "logprobs", "stop", "presence_penalty", "frequency_penalty", "best_of", "logit_bias",
                          "user"],
            response_value_keys=["choices", 0, "text"],
            llm_evaluator=None)
        if len(kwargs) > 0:
            confOpenAI = confOpenAI.combine(kwargs)
        return confOpenAI
    elif isinstance(spec, str) and spec.lower() == 'ChatGPT'.lower():
        confChatGPT = llm_configuration("openai",
                                        name="chatgpt",
                                        module='openai',
                                        model='gpt-3.5-turbo-0613',
                                        function=openai.chat.completions.create,  # was openai.ChatCompletion.create,
                                        known_params=["api_key", "model", "messages", "functions", "function_call",
                                                      "temperature", "top_p", "n",
                                                      "stream", "logprobs", "stop", "presence_penalty",
                                                      "frequency_penalty", "logit_bias",
                                                      "user"],
                                        response_value_keys=["choices", 0, "message", "content"])
        if len(kwargs) > 0:
            confChatGPT = confChatGPT.combine(kwargs)

        # Evaluator class
        confChatGPT.llm_evaluator = None

        return confChatGPT
    elif isinstance(spec, str) and spec.lower() == 'PaLM'.lower():

        # Set key
        apiKey = os.environ.get("PALM_API_KEY")
        apiKey = kwargs.get("api_key", apiKey)
        google.generativeai.configure(api_key=apiKey)

        # Configuration
        confPaLM = Configuration(
            name="palm",
            api_key=None,
            api_user_id="user",
            module="google.generativeai",
            model="models/text-bison-001",
            function=google.generativeai.generate_text,
            temperature=0.2,
            max_tokens=300,
            total_probability_cutoff=0.03,
            prompts=None,
            prompt_delimiter=" ",
            stop_tokens=None,
            argument_renames={"max_tokens": "max_output_tokens",
                              "stop_tokens": "stop_sequences"},
            fmt="values",
            known_params=[
                "model", "prompt", "temperature", "candidate_count", "max_output_tokens", "top_p", "top_k",
                "safety_settings", "stop_sequences", "client"
            ],
            response_object_attribute="result",
            response_value_keys=[],
            llm_evaluator=None)

        # Modify by additional arguments
        if len(kwargs) > 0:
            confPaLM = confPaLM.combine(kwargs)

        # Result
        return confPaLM

    elif isinstance(spec, str) and spec.lower() == 'ChatPaLM'.lower():

        # Start as PaLM text completion configuration
        confChatPaLM = llm_configuration("PaLM")

        # Default PaLM chat model
        confChatPaLM.model = 'models/chat-bison-001'

        # Function
        confChatPaLM.function = google.generativeai.chat

        # Get result
        # https://developers.generativeai.google/api/python/google/generativeai/types/ChatResponse
        confChatPaLM.response_object_attribute = "last"

        # The parameters are taken from here:
        #   https://github.com/google/generative-ai-python/blob/f370f5ab908a095282a0cdd946385db23c695498/google/generativeai/discuss.py#L210
        # and used in EvaluatorChatPaLM.eval
        confChatPaLM.known_params = [
            "model", "context", "examples", "temperature", "candidate_count", "top_p", "top_k", "prompt"
        ]

        # Adding it this for consistency
        confChatPaLM.response_value_keys = None

        # Evaluator class
        confChatPaLM.llm_evaluator = None

        # Combine with given additional parameters (if any)
        if len(kwargs) > 0:
            confChatPaLM = confChatPaLM.combine(kwargs)

        return confChatPaLM
    else:
        warnings.warn(f"Do not know what to do with given configuration spec: {spec}. Continuing with \"OpenAI\".")
        return llm_configuration('OpenAI', **kwargs)
    pass


# ===========================================================
# Evaluator creation
# ===========================================================
def llm_evaluator(spec, **args):
    # Default evaluator class
    evaluator_class = args.get('llm_evaluator_class', None)
    if evaluator_class is None:
        evaluator_class = Evaluator

    if evaluator_class is not Evaluator:
        raise ValueError(
            'The value of llm_evaluator_class is expected to be None or of type LLMFunctionObjects.Evaluator.')

    # Separate configuration from evaluator options
    attr_conf = list(llm_configuration('openai').to_dict().keys())
    args_conf = {k: v for k, v in args.items() if k in attr_conf}
    args_evlr = {k: v for k, v in args.items() if k not in args_conf and k not in ['llm_evaluator_class', 'form']}

    fd = {"formatron": args.get("formatron", args.get("form", None))}
    args_evlr = {**args_evlr, **fd}

    if spec is None:
        return Evaluator(conf=llm_configuration(None, **args_conf), **args_evlr)
    elif isinstance(spec, str):
        return llm_evaluator(llm_configuration(spec), **args_evlr,
                             llm_evaluator_class=args.get('llm_evaluator_class', None))
    elif isinstance(spec, Configuration):
        conf = spec.copy()
        if spec.llm_evaluator is None:
            return evaluator_class(conf=spec, **args_evlr)
        else:
            if not isinstance(conf.llm_evaluator, Evaluator):
                raise TypeError(
                    'The configuration attribute evaluator is expected' +
                    ' to be of type LLMFunctionObjects.Evaluator or None.')
            conf.llm_evaluator.conf = conf
            return conf.llm_evaluator
    elif isinstance(spec, Evaluator):
        res = spec.copy()
        conf = spec.conf.copy()

        if 'conf' in args_evlr:
            conf = llm_configuration(conf, **args_evlr['conf'])

        if args_conf:
            conf = llm_configuration(conf, **args_conf)

        res.conf = conf

        if 'formatron' in args_evlr:
            res.formatron = args_evlr['formatron']

        return res

    else:
        warnings.warn(
            "The first argument is expected to be None, or one of the types str," +
            " LLMFunctionObjects.Evaluator, or LLMFunctionObjects.Configuration.")
        return llm_evaluator(None, **args_evlr)


# ===========================================================
# Function creation
# ===========================================================
def llm_function(prompt='', **kwargs):
    llm_evaluator_spec = kwargs.get("llm_evaluator", kwargs.get("e", None))
    formatron_spec = kwargs.get("formatron", kwargs.get("form", kwargs.get("f", None)))
    llmEvaluator = llm_evaluator(spec=llm_evaluator_spec, formatron=formatron_spec)
    return Functor(llmEvaluator, prompt)


# ===========================================================
# Example function creation
# ===========================================================

def llm_example_function(rules, hint=None, **kwargs):
    hintLocal = hint
    if hintLocal is None:
        hintLocal = ""

    if isinstance(rules, dict):
        pre = ""
        for (k, v) in rules.items():
            pre = f"Input: {k}\nOutput: {v}\n\n"

        if isinstance(hintLocal, str) and len(hintLocal) > 0:
            hint = hint if re.search(r'.*{Punct}$', hint) else hint + '.'
            pre = f"{hint}\n\n{pre}"
        prompt = lambda x: pre + f"\nInput {x}\nOutput:"

        return llm_function(prompt, **kwargs)

    elif isinstance(rules, tuple) and len(rules) == 2:
        return llm_example_function({rules[0]: rules[1]}, **kwargs)

    elif isinstance(rules, list) and all(isinstance(x, tuple) and len(x) == 2 for x in rules):
        return llm_example_function({x[0]: x[1] for x in rules}, **kwargs)

    else:
        TypeError("The first argument is expected to be a tuple, a list of tuples, or a dictionary.")


# ===========================================================
# LLM Synthesise
# ===========================================================

def llm_synthesize(prompts, prop=None, **kwargs):
    # Get evaluator spec
    evlrSpec = kwargs.get("llm_evaluator", kwargs.get("e", None))

    # Prompts processing
    promptsLocal = prompts
    if isinstance(promptsLocal, str):
        promptsLocal = [promptsLocal, ]

    if not isinstance(promptsLocal, list):
        TypeError("The first argument is expected to be a string or list of string.")

    # Process properties
    expected_props = ['FullText', 'CompletionText', 'PromptText']

    if prop is None:
        prop = 'CompletionText'
    if not (isinstance(prop, str) and prop in expected_props):
        raise ValueError(
            f"The value of the second argument is expected to be Whatever or one of: {', '.join(expected_props)}.")

    # Get evaluator
    kwargs2 = {k:v for k,v in kwargs.items() if k not in ["e", "llm_evaluator"]}
    evalObj = llm_evaluator(evlrSpec, **kwargs2)

    # Add configuration prompts
    promptsLocal = evalObj.conf.prompts + promptsLocal
    evalObj.conf.prompts = []

    # Reduce prompts
    processed = []
    for p in promptsLocal:
        if isinstance(p, str):
            processed.append(p)
        elif callable(p):
            try:
                pres = p()
            except Exception:
                pres = None

            if not pres is not None or not isinstance(pres, str):
                args = [''] * p.__code__.co_argcount  # Get the arity of the function
                pres = p(*args)

            processed.append(pres)
        else:
            processed.append(str(p))

    # Find the separator from the configuration
    sep = evalObj.conf.prompt_delimiter
    prompt_text = sep.join(processed)

    # Post process
    if prop == 'FullText':
        res = llm_function('', e=evalObj)(prompt_text)
        return processed + [res]
    elif prop == 'PromptText':
        return prompt_text
    else:
        return llm_function('', e=evalObj)(prompt_text)


# ===========================================================
# Chat object creation
# ===========================================================

_mustPassConfKeys = ["name", "prompts", "examples", "temperature", "max_tokens",
                     "stop_tokens", "api_key", "api_user_id"]


def llm_chat(prompt: str = '', **kwargs):
    # Get evaluator spec
    spec = kwargs.get('llm_evaluator', kwargs.get('llm_configuration', kwargs.get('conf', None)))

    # Default evaluator class
    evaluator_class = kwargs.get('llm_evaluator_class', None)

    # Filter conf args
    conf_args = {k: v for k, v in kwargs.items() if k in list(llm_configuration(None).to_dict().keys())}

    if evaluator_class is not None and not isinstance(evaluator_class, EvaluatorChat):
        raise ValueError('The value of llm_evaluator_class is expected to be None or of the type EvaluatorChat.')

    # Make evaluator object
    if spec is None:
        # Make Configuration object
        conf = llm_configuration('ChatGPT', prompts=prompt, **conf_args)

        # Make Evaluator object
        llm_eval_obj = EvaluatorChatGPT(conf=conf, formatron=kwargs.get('form', kwargs.get('formatron')))

    elif isinstance(spec, Configuration) or isinstance(spec, Evaluator) or isinstance(spec, str):
        # Make Configuration object
        conf = llm_configuration(spec, prompts=prompt, **conf_args)

        # Obtain Evaluator class
        if evaluator_class is None:
            if 'palm' in conf.name.lower():
                conf = llm_configuration('ChatPaLM',
                                         **{k: v for k, v in conf.to_dict().items() if k in _mustPassConfKeys})
                evaluator_class = EvaluatorChatPaLM
            else:
                evaluator_class = EvaluatorChatGPT

        # Make Evaluator object
        llm_eval_obj = evaluator_class(conf=conf, formatron=kwargs.get('form', kwargs.get('formatron')))
    else:
        raise ValueError("Cannot obtain or make a LLM evaluator object with the given specs.")

    # Result
    args2 = {k: v for k, v in kwargs.items() if
             k not in ['llm_evaluator', 'llm_configuration', 'conf', 'prompt', 'form', 'formatron']}
    return Chat(llm_evaluator=llm_eval_obj, **args2)
