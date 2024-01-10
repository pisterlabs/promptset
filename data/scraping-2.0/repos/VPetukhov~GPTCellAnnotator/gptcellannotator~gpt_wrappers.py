import time
import openai
from tqdm.auto import tqdm
from pandas import Series, DataFrame
from typing import Optional


# CELL_TYPE_PROMPT = "Provide me a comprehensive list of cell types that are expected in {species} {tissue}. For each cell type provide {n_markers} main gene markers expressed in it. Write it in the format `- Type: gene_id1, gene_id2, ...` without any additional comments."
CELL_TYPE_PROMPT = "Provide me a comprehensive hierarchy of cell types that are expected in {species} {tissue}. Write it as an unordered list."
CELL_TYPE_MARKER_PROMPT = "Now, for each cell type specify a list of {n_markers} marker genes that are specific to it. Provide the answer as an unordered list without any additional comments. Example: `- Type X: gene_id1, gene_id2, ...`. Make sure that you provided markers for **each** cell type you mentioned above."

ANNOTATION_PROMPT = """
You need to annotate a {species} {tissue} dataset. You found gene markers for each cluster and need to determine clusters' identities.
Below is a short list of markers for each cluster:

{marker_list}

We expect the following cell types to be present in the dataset, however additionall types may also be present:
{expected_markers}

Determine cell type and state for cluster {cluster_id} (markers {cli_markers})
Only output data in the following format:
```
- Marker description: (description of what the markers mean. Example: markers A,B,C are associated with X, while D is related to Y)
- Cell type: (cell type name)
- Cell state: (cell state if there is anything specific, 'normal' otherwise)
- Confidence: one of 'low', 'medium', 'high'
- Reason: (reason for the confidence estimate)
```
""".strip()

AGENT_DESCRIPTION = "You're an expert bioinformatician, proficient in scRNA-seq analysis with background in {species} cell biology."


def _query_openai(
        agent_description: str, instruction: str, other_messages: Optional[list] = None,
        n_repeats: int = 4, model: str = "gpt-3.5-turbo", **kwargs
    ):
    if other_messages is None:
        other_messages = []

    for _ in range(n_repeats):
        res = None
        try:
            res = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": agent_description},
                    {"role": "user", "content": instruction}
                ] + other_messages,
                **kwargs
            )
            break
        except openai.OpenAIError as e:
            print(e)
            time.sleep(1)
            continue

    if res is None:
        raise Exception("Failed to get response from OpenAI")

    return res


def get_expected_cell_types(species: str, tissue: str, n_markers: int = 5, verbose: bool = True, **kwargs):
    prompt = CELL_TYPE_PROMPT.format(species=species, tissue=tissue)
    agent_desc = f"You're an expert in {species} cell biology."

    if verbose:
        print("Querying cell types...")

    res_types = _query_openai(agent_desc, agent_desc + ' ' + prompt, **kwargs)
    expected_types = res_types.choices[0].message.content

    prompts2 = [
        {"role": "assistant", "content": expected_types},
        {"role": "user", "content": CELL_TYPE_MARKER_PROMPT.format(n_markers=n_markers)}
    ]

    if verbose:
        print("Querying cell type markers...")

    res_markers = _query_openai(agent_desc, agent_desc + ' ' + prompt, prompts2, **kwargs)
    expected_markers = res_markers.choices[0].message.content

    if verbose:
        print("Done!")

    return expected_types, expected_markers


def annotate_clusters(
        marker_genes: Series, species: str, tissue: str,
        expected_markers: Optional[str] = None, annotation_prompt: str = ANNOTATION_PROMPT, **kwargs
    ):
    answers = {}
    if expected_markers is None:
        expected_markers = get_expected_cell_types(species=species, tissue=tissue, model='gpt-4', max_tokens=800)[1]

    marker_txt = "\n".join([f'- Cluster {i}: {", ".join(gs)}' for i,gs in marker_genes.items()])
    agent_desc = AGENT_DESCRIPTION.format(species=species)

    for cli in tqdm(marker_genes.index):
        cli_markers = ", ".join(marker_genes[cli])

        pf = annotation_prompt.format(
            species=species, tissue=tissue, marker_list=marker_txt,
            cluster_id=cli, cli_markers=cli_markers, expected_markers=expected_markers
        )

        res = _query_openai(agent_desc, pf, **kwargs)
        answers[cli] = res.choices[0].message.content

    return answers


def parse_annotation(annotation_res: str):
    ann_df = DataFrame(
        Series(annotation_res).map(
            lambda ann: dict([l.strip().lstrip('- ').split(': ') for l in ann.split('\n') if len(l.strip()) > 0])
        ).to_dict()
    ).T[['Marker description', 'Cell type', 'Cell state', 'Confidence', 'Reason']]

    ann_df['Cell type, raw'] = ann_df['Cell type']
    ann_df['Cell type'] = ann_df['Cell type'].map(
        lambda x: x.replace('cells', '').replace('cell', '').replace('.', '').strip().split('(')[0].split(',')[0].strip().replace('  ', ' ')
    )

    ann_df['Confidence'] = ann_df['Confidence'].str.rstrip('.').str.lower()

    return ann_df
