import openai

from datasets import load_metric

from fp_dataset_artifacts.utils import init_openai, get_finetune_response
from fp_dataset_artifacts.snli import map_refs_and_preds


INTRO = (
    'I am a highly intelligent natural language inference bot. '
    + 'If you give me a pair of premise and hypothesis, '
    + 'I will tell you whether the hypothesis is entailment, neutral, or contradiction.'
)


def int2label(i):
    return ['Entailment', 'Neutral', 'Contradiction'][i]


def label2int(x):
    return {'Entailment': 0, 'Neutral': 1, 'Contradiction': 2}[x]


def map_finetune(x):
    premise = x['premise']
    hypothesis = x['hypothesis']
    label = int2label(x['label'])

    return {
        'prompt': f"Premise: {premise}\n\nHypothesis: {hypothesis}\n\nLabel: ",
        'completion': label
        + '\n',  # '\n' added to prevent further text generation.
    }


def get_examples(exs):
    prompts = exs['prompt']
    completions = exs['completion']
    examples = [p + c for p, c in zip(prompts, completions)]
    return '\n\n'.join(examples)


def get_prompt(x, exs, intro=INTRO):
    examples = get_examples(exs)
    query = x['prompt']
    prompt = f'{intro}\n\n{examples}\n\n{query}'
    return prompt


def get_response(prompt, model):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0.3,
        max_tokens=10,
        stop=['\n'],
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0,
    )
    return response


def finetune(
    train_file_id,
    valid_file_id,
    model='curie',
    n_epochs=4,
    compute_classification_metrics=True,
    classification_n_classes=3,
):
    # Initialize OpenAI API with API_KEY
    init_openai()

    # Create fine-tune
    finetune_resp = openai.FineTune.create(
        training_file=train_file_id,
        validation_file=valid_file_id,
        model=model,
        n_epochs=n_epochs,
        compute_classification_metrics=compute_classification_metrics,
        classification_n_classes=classification_n_classes,
    )

    # Get the fine-tune ID
    finetune_id = finetune_resp['id']

    return finetune_id


def evaluate(
    finetune_id,
    test,
    responses_local_filename,
):
    # Initialize OpenAI API with API_KEY
    init_openai()

    # Check if fine-tuning has completed.
    # And retrieve the model name.
    finetune_resp = openai.FineTune.retrieve(finetune_id)

    assert (
        finetune_resp['events'][-1]['message'] == 'Fine-tune succeeded'
    ), 'Please wait for the fine-tuning to be completed.'

    # Get the model name
    model = finetune_resp['fine_tuned_model']

    # Evaluate the model on test set
    def map_response(x):
        try:
            response = get_finetune_response(x['prompt'], model)
            return {'response': response['choices'][0]['text']}
        except Exception as e:
            return {'response': None}

    responses = test.map(map_response)

    # Save the responses
    responses.to_json(f'../results/{responses_local_filename}')

    # Load metrics
    f1_metric = load_metric('f1')
    acc_metric = load_metric('accuracy')

    # Convert response to references and predictions for metrics
    results = responses.map(map_refs_and_preds)

    # Compute metrics
    f1 = f1_metric.compute(
        references=results['references'],
        predictions=results['predictions'],
        average='weighted',
    )

    accuracy = acc_metric.compute(
        references=results['references'], predictions=results['predictions']
    )

    print(f'{f1=}')
    print(f'{accuracy=}')

    return model
