def main(api_key: str, pubmed_email: str, use_pickled_config, random_seed: int | None):
    from run_classifier.run_classifier import RunClassifier
    import asyncio

    import dill as pickle
    import seaborn as sns
    import openai

    import random
    import numpy as np

    sns.set_theme(style="darkgrid")
    openai.api_key = api_key

    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)

    if use_pickled_config:
        with open(use_pickled_config, "rb") as f:
            config = pickle.load(f)
            asyncio.run(config.run_all())
    else:
        run = RunClassifier(pubmed_email)
        asyncio.run(run.run())


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run baseline protect model.")
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="open api key",
    )
    parser.add_argument(
        "--pubmed_email",
        type=str,
        required=True,
        help="email to use for pubmed sources",
    )
    parser.add_argument(
        "--use_pickled_config",
        type=Path,
        help="whether to use saved object",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="random seed",
    )

    args = parser.parse_args()

    main(args.api_key, args.pubmed_email, args.use_pickled_config, args.random_seed)
