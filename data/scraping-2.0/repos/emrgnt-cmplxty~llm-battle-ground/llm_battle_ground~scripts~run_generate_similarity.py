import os

import openai
from evalplus.data import write_jsonl

from llm_battle_ground.helpers import SimilarityExperimentRunner
from llm_battle_ground.scripts import common_arg_parser
from llm_battle_ground.utils import get_configured_logger

# Local constants
NUM_INPUT_EXAMPLES = 20
NUM_OUTPUT_EXAMPLES = 20
STEP_SIZE = 40
BUFFER = 10
PROVIDER = "openai"

if __name__ == "__main__":
    # Initialization
    parser = common_arg_parser()
    parser.add_argument(
        "--num-input-examples",
        type=int,
        default=NUM_INPUT_EXAMPLES,
        help="Number of preceeding examples to include in N-shot context.",
    )
    parser.add_argument(
        "--num-output-examples",
        type=int,
        default=NUM_OUTPUT_EXAMPLES,
        help="Number of preceeding examples to include in N-shot context.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=STEP_SIZE,
        help="Iteration step size when running the experiment",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=BUFFER,
        help="The size of the forward example buffer",
    )
    parser.add_argument(
        "--perplexity",
        action="store_true",
        help="Whether to use perplexity as a measure of similarity (note can only do this with local models)",
    )

    args = parser.parse_args()

    provider = args.provider or PROVIDER
    if (
        provider == "openai"
        or provider in ["hugging-face"]
        and not args.perplexity
    ):
        # TODO - Rename this to `OPENAI_API_KEY`
        openai.api_key = os.getenv("OPENAI_API_KEY_LOCAL", "")

    assert not (
        provider == "openai" and args.perplexity
    ), "Cannot use perplexity with OpenAI API"

    logger = get_configured_logger(__name__, args.log_level)

    # Create the experiment and run it
    experiment = SimilarityExperimentRunner(logger, args)
    outputs = experiment.run()

    # Finalize
    write_jsonl(experiment.out_path, outputs)
