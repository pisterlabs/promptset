import glob
import json
import logging
from pathlib import Path

import numpy as np
import openai
from openai.embeddings_utils import get_embedding
from pandas import DataFrame
from pandas import Series

from chart_gpt.schemas import ChartGptModel
from chart_gpt.utils import extract_json
from chart_gpt.utils import generate_completion
from chart_gpt.utils import json_dumps_default
from chart_gpt.utils import pd_vss_lookup

logger = logging.getLogger(__name__)

CHART_CONTEXT_EXCLUDE_EXAMPLES = [
    "layer_likert",  # 4532
    "isotype_bar_chart",  # 3924
    "interactive_dashboard_europe_pop",  # 3631
    "layer_line_window"  # 3520
]
CHART_DEFAULT_CONTEXT_ROW_LIMIT = 5
VEGA_LITE_CHART_PROMPT_FORMAT = """
Examples:
{examples}
Generate a Vega Lite chart that answers the user's question from the data.
User question: {question}
Tabular dataset (these will be automatically included in data.values):
{result}
Vega-Lite definition following schema at https://vega.github.io/schema/vega-lite/v5.json:
"""
CHART_EMBEDDING_MODEL = 'text-embedding-ada-002'


class ChartIndex(ChartGptModel):
    # [chart_id] -> [specification]
    specifications: DataFrame
    # [chart_id] -> [dim_0, dim_1, ..., dim_n]
    embeddings: DataFrame

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(cls) -> "ChartIndex":
        root_dir = Path(__file__).parent
        vl_example_files = glob.glob('data/vega-lite/examples/*.vl.json',
                                     root_dir=root_dir)
        vl_example_names = [
            example.removeprefix('data/vega-lite/examples/').removesuffix('.vl.json')
            for example in vl_example_files
        ]
        vl_example_specs_json = [open(root_dir / example, 'r').read() for example in vl_example_files]
        response = openai.Embedding.create(
            input=vl_example_specs_json,
            engine=CHART_EMBEDDING_MODEL
        )
        embeddings = [item.embedding for item in response.data]
        embeddings_df = DataFrame(
            embeddings,
            index=vl_example_names
        ).rename(lambda i: f"dim_{i}", axis=1).drop(CHART_CONTEXT_EXCLUDE_EXAMPLES)
        specifications_df = DataFrame(
            vl_example_specs_json,
            index=vl_example_names,
            columns=['specification']
        ).drop(CHART_CONTEXT_EXCLUDE_EXAMPLES)
        return ChartIndex(embeddings=embeddings_df, specifications=specifications_df)

    def top_charts(self, question: str, data: DataFrame) -> Series:
        """
        Finds the most relevant chart specifications to the given question and data.
        :return: Series [chart_id] -> specification
        """
        logger.info("Finding examples for question: %s", question)
        embedding_query_string = json.dumps(
            {
                "title": question,
                "data": {
                    "values": data.head(CHART_DEFAULT_CONTEXT_ROW_LIMIT).to_dict(orient="records")
                }
            },
            default=json_dumps_default
        )
        embedding_query = np.array(
            get_embedding(embedding_query_string, engine=CHART_EMBEDDING_MODEL)
        )
        chart_ids = pd_vss_lookup(self.embeddings, embedding_query, n=3).index.tolist()
        examples = self.specifications.specification.loc[chart_ids].map(json.loads).tolist()
        logger.info("Found examples: %s", examples)
        return examples


class ChartGenerator(ChartGptModel):
    index: ChartIndex

    def generate(self, question: str, result_set: DataFrame) -> dict:
        logger.info("Generating chart for question: %s", question)
        data_values = result_set.to_dict(orient='records')
        top_charts = self.index.top_charts(question, result_set)
        sample_data = data_values[:CHART_DEFAULT_CONTEXT_ROW_LIMIT]
        prompt = VEGA_LITE_CHART_PROMPT_FORMAT.format(
            result=json.dumps(sample_data, default=json_dumps_default),
            question=question,
            examples=json.dumps(top_charts, default=json_dumps_default)
        )
        completion = generate_completion(prompt, temperature=0.)
        specification = extract_json(completion)
        specification['data'] = {'values': data_values}
        logger.info("Generated chart %s", specification)
        return specification
