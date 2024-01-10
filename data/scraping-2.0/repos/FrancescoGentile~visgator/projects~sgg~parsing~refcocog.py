##
##
##

import asyncio
import json
import pickle
from pathlib import Path

from openai import error
from tqdm import tqdm

from datasets.refcocog import Config, Dataset
from deepsight.data.dataset import Split
from deepsight.data.structs import Entity, SceneGraph, Triplet
from deepsight.modeling.parsers.gpt import SceneGraphParser

example1 = (
    "There is a truck covered in snow farthest from the right",
    SceneGraph.new(
        entities=[
            Entity("truck", "a truck farthest from the right"),
            Entity("snow", "snow"),
        ],
        triplets=[
            Triplet(0, "covered in", 1),
        ],
    ),
)

example2 = (
    "A placemat is empty behind a placemat that is full",
    SceneGraph.new(
        entities=[
            Entity("placemat", "an empty placemat"),
            Entity("placemat", "a full placemat"),
        ],
        triplets=[
            Triplet(0, "behind", 1),
        ],
    ),
)

example3 = (
    "the chair not being used in the background, perpendicular to the viewer",
    SceneGraph.new(
        entities=[
            Entity("chair", "the chair not being used in the background"),
            Entity("viewer", "the viewer"),
        ],
        triplets=[
            Triplet(0, "perpendicular to", 1),
        ],
    ),
)

example4 = (
    "A double decker bus with the wording The Ghost Bus Tours.com on the side.",
    SceneGraph.new(
        entities=[
            Entity("bus", "a double decker bus"),
            Entity("wording", "the wording The Ghost Bus Tours.com"),
            Entity("side", "the side"),
        ],
        triplets=[
            Triplet(0, "with", 1),
            Triplet(1, "on", 2),
        ],
    ),
)

example5 = (
    "a man stands majestically on his skis on a snow covered area with 2 other people "
    + "behind him in the distance",
    SceneGraph.new(
        entities=[
            Entity("man", "a man"),
            Entity("ski", "his skis"),
            Entity("snow area", "a snow covered area"),
            Entity("people", "2 other people"),
        ],
        triplets=[
            Triplet(0, "stands on", 1),
            Triplet(0, "on", 2),
            Triplet(3, "behind", 0),
        ],
    ),
)

example6 = (
    "Woman in white shirt looking down at laptop computer and " + "holding a glass",
    SceneGraph.new(
        entities=[
            Entity("woman", "woman"),
            Entity("shirt", "white shirt"),
            Entity("computer", "laptop computer"),
            Entity("glass", "a glass"),
        ],
        triplets=[
            Triplet(0, "in", 1),
            Triplet(0, "looking down at", 2),
            Triplet(0, "holding", 3),
        ],
    ),
)

example7 = (
    "Woman on the right",
    SceneGraph.new(entities=[Entity("woman", "woman on the right")], triplets=[]),
)

examples = [example1, example2, example3, example4, example5, example6, example7]


class Generator:
    def __init__(
        self,
        batch_size: int = 10,
        rpm: int = 10,
        max_retries: int = 3,
    ) -> None:
        self.batch_size = batch_size
        self.rpm = rpm
        self.max_retries = max_retries

        self.parser = SceneGraphParser(api_key="")

    async def process_batch(
        self, batch: list[str]
    ) -> list[tuple[str, SceneGraph | None]]:
        num_retries = 0
        delay = 16
        while num_retries < self.max_retries:
            try:
                res = await asyncio.wait_for(
                    self.parser.parse(examples, batch), timeout=60
                )
                return res
            except error.RateLimitError:
                print("Rate limit exceeded")
                num_retries += 1
                delay *= 2
                await asyncio.sleep(delay)
            except error.ServiceUnavailableError:
                print("Service unavailable")
                num_retries += 1
                delay *= 2
                await asyncio.sleep(delay)
            except error.APIError:
                print("API error")
                num_retries += 1
                delay *= 2
                await asyncio.sleep(delay)
            except TimeoutError:
                print("Timeout error")
                num_retries += 1
                delay *= 2
                await asyncio.sleep(delay)
            except Exception as e:
                raise e

        raise Exception("Failed to parse batch")

    async def run(self) -> None:
        config = Config(Path("data/refcocog"))
        splits = [Split.TEST]

        errors_file = Path("data/refcocog/errors.json")
        output_file = Path("data/refcocog/scene_graphs.json")
        if output_file.exists():
            with output_file.open("r") as f:
                data = json.load(f)
        else:
            data = {}

        if errors_file.exists():
            with errors_file.open("rb") as f:
                errors = pickle.load(f)
        else:
            errors = set()

        for split in splits:
            dataset = Dataset.new_for_rec(config, split, False)
            samples = dataset._samples
            progress_bar = tqdm(
                total=len(samples), desc=f"Processing {split.name}", leave=False
            )

            last_idx = 0
            idx = 0

            while idx < len(samples):
                promises = []

                # start = time.time()
                for _ in range(self.rpm):
                    batch = []

                    count = 0
                    while count < self.batch_size:
                        if idx >= len(samples):
                            break

                        description = samples[idx].description
                        if description not in data:
                            count += 1
                            batch.append(samples[idx].description)

                        idx += 1

                    promises.append(self.process_batch(batch))

                    if len(batch) < self.batch_size:
                        break

                if len(promises) == 0:
                    break

                results: list[list[tuple[str, SceneGraph | None]]]
                results = await asyncio.gather(*promises)
                promises.clear()
                # end = time.time()

                for result in results:
                    for description, scene_graph in result:
                        if scene_graph is not None:
                            data[description] = scene_graph.to_dict()
                            if description in errors:
                                errors.remove(description)
                        elif description not in errors:
                            errors.add(description)

                tqdm.update(progress_bar, idx - last_idx)
                last_idx = idx

                with output_file.open("w") as f:
                    json.dump(data, f)

                with errors_file.open("wb") as f:
                    pickle.dump(errors, f)

                # if end - start < 60:
                #    time.sleep(60 - (end - start))

            progress_bar.close()


if __name__ == "__main__":
    generator = Generator()
    asyncio.run(generator.run())
