import os
import openai
import json
import spacy

nlp = spacy.load("en_core_web_sm")

GENERATE_SYSTEM_PROMPT = \
"""You are an educational assistant capable of classifying test questions
into several different categories within a given an area of study.

You will receive questions as JSON where each key is a test question
and each value is a list sorted by category specificity. The list will always be
initialized to contain the area of study the question belongs to, potentially
proceeded by categories, in increasing specificity, within that discipline
the question belongs to. An example of a dictionary
you might receive is as follows:

```
{
    'What is known as the powerhouse of the cell?': ['biology'],
    'What is the part of the cell that contains genetic information?': ['biology'],
    'What is a good definition of overfitting?': ['machine learning']
}
```

Note that the spacing may not be uniform like it is written here.
Then, you will output a dictionary where each value (each list) has exactly one extra category
appended to it. The new category must be highly correlated with the question.
In general, to produce the output, you will use the following steps:

Step 1: For each question, identify the corresponding value, which will always be a list, and
observe the first and last element of that list. The first element will always be
the area of study the question belongs to. The last element will be the most specific categorization
of the question the user has provided. So, the last element may either also be the area of study
the question belongs to, or a category within the area of study the question belongs to.

Step 2: Using the question text, the area of study the question belongs to (first element of the value), and the
most specific categorization of the question the user provided (last element of the value), generate
a new category that meets the following criteria:
    - The new category is more specific that the most specific categorization of the question the
    user provided.
    - The new category is as general as possible.

After this step, for the example input, your output might look like this:

```
{
    "What is known as the powerhouse of the cell?": ["biology", "parts of the cell"],
    "What is the part of the cell that contains genetic information?": ["biology", "organelles"],
    "What is mRNA?": ["biology", "genetics"],
    "What is a good definition of overfitting?": ["machine learning", "model training"]
}
```

Step 3: For each area of study in the input, observe the categories you appended. Any categories
that are too similar must be combined into one. For example, in the example output from step 2,
'parts of the cell', 'organelles', and 'genetics' are the new categories you added for the 'biology'
area of study. Since 'parts of the cell' and 'organelles' are quite similar, you should combine
them into one. That is, you ensure that only 'parts of the cell' or 'organelles' is used
for the questions whose lists had either 'parts of the cell' or 'organelles' appended.
Alternatively, similarity in categories suggests there exists a category that is more general
that my describe both. For example, one could use 'cell biology' to encapsulate
'organelles' and 'parts of the cell'. Note in this example that
'genetics' is sufficiently distinct from the other two, so it does not have to change.

After this final step, you might have produced something that looks like this:

```
{
    "What is known as the powerhouse of the cell?": ["biology", "cell biology"],
    "What is the part of the cell that contains genetic information?": ["biology", "cell biology"],
    "What is mRNA?": ["biology", "genetics"],
    "What is a good definition of overfitting?": ["machine learning", "model training"]
}
```

Note you also output in JSON form."""

def remove_duplicate_categories(result_dict):
    most_recent_categories = {}

    for v in result_dict.values():
        if v[0] not in most_recent_categories.keys():
            most_recent_categories[v[0]] = [nlp(v[-1])]
        else:
            emb1 = nlp(v[-1])
            for emb2 in most_recent_categories[v[0]]:
                if emb1.similarity(emb2) > 0.5:
                    continue

            most_recent_categories[v[0]].append(emb1)

    most_recent_categories = list(most_recent_categories)
    return most_recent_categories

def generate_category(questions_dict):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    dicts = []
    if len(questions_dict) > 4000:
        for i in range(0, 4000, 400):
            dicts.append(dict((k, v) for k, v in questions_dict.items()[i:i + 400]))
    else:
        dicts = [questions_dict]

    completions = []
    for q_dict in dicts:
        completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
                            {"role": "user", "content": str(q_dict)}
                        ]
                    )

        completion = json.loads(completion["choices"][0]["message"]["content"])

        for key in completion.keys():
            completion[key] = completion[key][:len(questions_dict[key]) + 1]

        completions.append(completion)

    output = {}
    for comp in completions:
        output = output | comp

    return output

example_questions = {
    "What is the limit of cos(x)/x as x->0?": ["calculus"],
    "What is the limit of e^{-x} as x->\infty?": ["calculus"],
    "Write e^x using an infinite sum": ["calculus"],
    "What is the precision of two-point Gaussian quadrature?": ["numerical computing"],
    "Why does Q-learning work, even though it is a biased method?": ["machine learning"],
    "What is Temporal Difference Learning in mathematical terms?": ["machine learning"],
    "Prove that the integral of 1/n does not converge.": ["calculus", "integration"],
    "Write the equation to find the price that will be set by a monopoly.": ["economics"],
    "Why is marginal revenue not equal to price for a monopoly?": ["economics"],
    "Write the general equation to find market supply.": ["economics"],
    "What is Nash equilibrium?": ["economics"]
}

example_questions2 = {
    "Osmosis - The movement of water between permeable membranes": ["biology"],
    "Diffusion - The movement of particles throug permeable membranes": ["biology"],
}

if __name__ == "__main__":
    print(generate_category(example_questions))