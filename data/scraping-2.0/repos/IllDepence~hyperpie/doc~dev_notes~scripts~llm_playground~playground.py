import datetime
import json
import openai


def prompt_and_save(para, prompt, params, o_ai):
    """ Convenience method that
        - gets completion from OpenAI API
        - saves result to a JSON file
        - echoes the prompt
        - echoes the completion
        - returns the completion
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fn = f"prompt_result_{timestamp}.json"

    completion = o_ai.Completion.create(prompt=prompt, **params)
    out_dict = {
        "paragraph": para,
        "timestamp": timestamp,
        "prompt": prompt,
        "params": params,
        "completion": completion
    }

    with open(out_fn, "w") as f:
        json.dump(out_dict, f, indent=4)

    print('- - - Prompt - - -')
    print(prompt)
    print('- - - Completion - - -')
    print(out_dict["completion"]["choices"][0]["text"])
    print('- - - Saved to - - -')
    print(out_fn)

    return out_dict["completion"]


with open('api_organization') as f:
    openai.organization = f.read().strip()
with open('api_key') as f:
    openai.api_key = f.read().strip()
# openai.api_base = "http://localhost:5000/v1"
# openai.Model.list()

# Instruction texts
instruction_context = "In the context of machine learning and related fields,"
instruction_yn_question = "Output only either \"yes\" or \"no\"."  # noqa: E501
instruction_entity = "Entity (a model/method/dataset)"
instruction_output = "Output:\n"

text_has_artifs = f"""{instruction_context} does the Input Text below mention any model/method/dataset?

[Input Text start]
{{text}}
[Input Text end]

{instruction_yn_question}

{instruction_output}"""  # noqa: E501

artif_is_out_of_scope = f"""{instruction_context} does the {instruction_entity} below fall into any of the following categories?

- a machine learning library
- a machine learning framework
- a machine learning task
- a metric
- an API

Entity: {{entity}}

{instruction_yn_question}

{instruction_output}"""  # noqa: E501

text_which_artifs = f"""{instruction_context} what are the entities (dataset/model/method/loss function/regularization technique) mentioned in the Input Text below?

[Input Text start]
{{text}}
[Input Text end]

Answer in the following YAML format.

Format:
---
- entity<N>:
    name: <entity name>
    type: <entity type>
...

Only produce output in the YAML format specified above. Output no additional text.

{instruction_output}"""  # noqa: E501

artif_has_paras = f"""{instruction_context} does the following {instruction_entity} have parameters for which a value can be chosen?

Entity: {{entity}}

{instruction_yn_question}

{instruction_output}"""  # noqa: E501

artif_which_paras = f"""{instruction_context} what are the parameter(s) of the following {instruction_entity}, for which a value can be chosen when using it?

Entity: {{entity}}

For each parameter, name its formula symbol (if it has one), as well as typical values of the parameter. Answer in the following YAML format.

Format:
---
- parameter<N>:
    name: <parameter name>
    formula symbol: <formula symbol>/null
    typical values: <typical values>
...

only produce output in the YAML format specified above. Output no additional text.

{instruction_output}"""  # noqa: E501

text_has_values = f"""{instruction_context} does the Input Text below menion any numerical quantities?

[Input Text start]
{{text}}
[Input Text end]

{instruction_yn_question}

{instruction_output}"""  # noqa: E501

text_which_relations = f"""{instruction_context} we consider the following list of Entities.

Entities:
{{entities}}

What (if any) are parameters and values of the Entities above described in the Input Text below?

[Input Text start]
{{text}}
[Input Text end]

For each entitiy, list its parameter(s) and value(s), if any, in in the following YAML format.

Format:
---
- entity<N>:
    name: <entity name>
    has_parameters: true/false
    parameters:
        - parameter<N>:
            name: <parameter name>
            value: <parameter value>/null
            context: <value context>/null
...

Only produce output in the YAML format specified above. Output no additional text.

{instruction_output}"""  # noqa: E501

text_e2e = f"""{instruction_context} what (if any) are the entities (datasets, models, methods, loss functions, regularization techniques) mentioned in the LaTeX Input Text below? What (if any) are their parameters and values?

[LaTeX Input Text start]
{{text}}
[LaTeX Input Text end]

Answer in the following YAML format.

Format:
---
- text_contains_entities: true/false
- entities (datasets, models, methods, loss functions, regularization techniques):
    - entity<N>:
        name: <entity name>
        type: <entity type>
        has_parameters: true/false
        parameters:
            - parameter<N>:
                name: <parameter name>
                value: <parameter value>/null
                context: <value context>/null
...

Only produce output in the YAML format specified above. Output no additional text.

{instruction_output}"""  # noqa: E501

params = {
    "model": "text-davinci-003",  # "default" for other models
    "max_tokens": 512,
    "temperature": 0.0,           # 0 - 2
    "top_p": 1,                   # default 1, change only w/ detault temp
    "n": 1,                       # num completions to generate
    "logprobs": 0,                # return log probs of n tokens (max 5)
    "echo": False,
}

# TODO: add test with % too se if GPT can deal with it

demo_text_positive_1 = """We divide our 438 annotated documents into training (70%), validation (30%) and test set (30%). The base document representation of our model is formed by SciBERT-base [4] and BiLSTM with 128-d hidden state. We use a dropout of 0.2 after BiLSTM embeddings. All feedforward networks are composed of two hidden layers, each of dimension 128 with gelu activation and with a dropout of 0.2 between layers. For additive attention layer in span representation, we collapse the token embeddings to scalars by passing through the feedforward layer with 128-d hidden state and performing a softmax. We train our model for 30 epochs using Adam optimizer with 1e-3 as learning rate for all non BERT weights and 2e-5 for BERT weights. We use early stopping with a patience value of 7 on the validation set using relation extraction F1 score. All our models were trained using 48Gb Quadro RTX 8000 GPUs. The multitask model takes approximately 3"""  # noqa: E501

demo_text_positive_2 = """Our system extends the implementation and hyper-parameters from Lee2017EndtoendNC with the following adjustments. We use a 1 layer BiLSTM with 200-dimensional hidden layers. All the FFNNs have 2 hidden layers of 150 dimensions each. We use 0.4 variational dropout [15] for the LSTMs, 0.4 dropout for the FFNNs, and 0.5 dropout for the input embeddings. We model spans up to 8 words. For beam pruning, we use \(\lambda _{\\text{C}}=0.3\) for coreference resolution and \(\lambda _{\\text{R}}=0.4\) for relation extraction. For constructing the knowledge graph"""  # noqa: E501

demo_text_norel = """Results in Table REF show that we perform generally better than DyGIE++. The performance on end-to-end binary relations shows the utility of incorporating a document level model for cross-section relations, rather than predicting on individual sections. Specifically, We observe a large difference in recall, which agrees with the fact that 55% of binary relation occur across sentence level. DyGIE++ (All sections) were not able to identify any binary relations because 80% of training examples have no sentence level binary relations, pushing the model towards predicting very few relations. In contrast, training on SciERC (and evaluating on SciREX) gives better results because it is still able to find the few sentence-level relations."""  # noqa: E501

demo_text_nothing = """To measure if automatic labeling is making the human annotation faster, we also asked our annotator to perform annotations on five documents without automatic labeling. We compute the difference in time between these two forms of annotation per entity annotated. Note that here, we only ask our annotator to annotate salient mentions. With the automatic labeling, annotation speed is 1.34 sec per entity time vs. 2.48 sec per entity time on documents without automatic labeling (a 1.85x speedup). We also observe 24% improvement in recall of salient mentions by including non-salient mentions, further showing the utility of this approach. """  # noqa: E501

demo_text_nothing2 = """In this paper, we investigate if preprints are affected by citation bias concerning the author affiliation. We focus on the author affiliation, as a survey by Soderberg et al. [32] observed that 35% of respondents consider the author's institution as extremely or very important to assess the credibility of preprints. Therefore, we assume that author affiliation has an influence on the citation counts of preprints. We verify the existence of citation bias by computing citation inequality. To this end, we measure to which degree the number of citations that preprints and their publisher versions receive is unequally distributed. Specifically, we measure citation bias with regard to author affiliation on the institution level and country level. Comparing differences in the citation inequality between preprints and their respective publisher versions allows us to mitigate the effects of confounding factors and see whether or not citation biases related to author affiliation have an increased effect on preprint citations. Conclusions drawn from this type of investigation are based on the assumption that the process of peer-review and formal publication is generally perceived as an assurance of quality [25] and therefore “levels the playing field” among articles in terms of citability. """  # noqa: E501

# completion = openai.Completion.create(prompt=prompt, **params)
