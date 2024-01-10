import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

from logic import generate_transformation_functions
from processor import DataFrameProcessor
from utils.logging import get_basic_stdout_logger

if __name__ == '__main__':
    logger = get_basic_stdout_logger()

    if not os.getenv('OPENAI_API_KEY'):
        logger.info("OPENAI_API_KEY not found in environment variables. Loading from .env file")
        load_dotenv()

    parser = argparse.ArgumentParser(description='Convert source CSV based on template and save to target CSV.')
    parser.add_argument('-s', '--source', type=str, required=True, help='Path to the source CSV file.')
    parser.add_argument('-t', '--template', type=str, required=True, help='Path to the template CSV file.')
    parser.add_argument('-o', '--target', type=str, default='output.csv',
                        help='Path to the target CSV file where results will be saved.')
    parser.add_argument('--separator', type=str, default=',', help='Separator used in the CSV files.')
    parser.add_argument('-d', '--depth', type=int, default=5,
                        help='Count of sample values to use for type inference. Default: 5')

    args = parser.parse_args()

    fname_template_table = args.template
    fname_source_table = args.source
    fname_target_table = args.target
    separator = args.separator
    depth = args.depth

    template_table = pd.read_table(fname_template_table, sep=separator)
    source_table = pd.read_table(fname_source_table, sep=separator)

    llm = ChatOpenAI(model=os.getenv('OPENAI_MODEL_NAME'), temperature=os.getenv('OPENAI_TEMPERATURE'))
    transformations, raw = generate_transformation_functions(llm=llm, template_table=template_table,
                                                             source_table=source_table, n_rows=depth)
    processor = DataFrameProcessor.from_dict(transformations=transformations)

    output = processor(source_table)
    output.to_csv(fname_target_table, index=False)
