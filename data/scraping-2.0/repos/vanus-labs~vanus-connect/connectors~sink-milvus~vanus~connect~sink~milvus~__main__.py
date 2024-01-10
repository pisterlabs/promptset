# SPDX-FileCopyrightText: 2023 Linkall Inc.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

import aiorun
import yaml
from vanus.connect.cdk import build_pipeline
from vanus.connect.cloudevents import CloudEventSource
from vanus.connect.service.openai import OpenAIEmbeddingService

from .sink import MilvusSink


def main():
    parser = argparse.ArgumentParser(prog="milvus-sink", description="vanus connect sink milvus", epilog="Linkall Inc.")
    parser.add_argument("--config", help="the path of configuration file")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = dict()

    pipeline = build_pipeline(CloudEventSource(config.get("port", 8080)))
    if config.get("embedding", True):
        pipeline = pipeline.call(OpenAIEmbeddingService(**config["openai_embedding"]))
    pipeline = pipeline.then(MilvusSink(**config["milvus"]))

    aiorun.run(pipeline.start())


if __name__ == "__main__":
    main()
