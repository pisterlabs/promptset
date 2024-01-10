import argparse
import asyncio
import json

import openai
from pymilvus import Collection

from app.config import load_config
from app.embedding import EMBEDDING_MODEL
from app.schemas.ads import collection_name, schema
from app.scripts.create_collection_ads import connect_milvus, disconnect_milvus

AD_EMBEDDING_FIELDS = [
  "title",
  "description",
]


def connect_openapi():
  config = load_config()
  openai.api_key = config['openai_secret']


def parse_dump(file_path: str) -> list[dict]:
  with open(file_path, "r") as f:
    data = json.load(f)

  data_points = []
  for item in data:
    data_point = {"id": item["pk"]}
    for field_name, field_value in item["fields"].items():
      if field_name in AD_EMBEDDING_FIELDS:
        data_point[field_name] = field_value
        data_points.append(data_point)

  return data_points


async def prepare_ad(project_name: str, raw_ad: dict):
  text = ""
  for field_name, field_value in raw_ad.items():
    if field_name in AD_EMBEDDING_FIELDS:
      text += str(field_value)

  embeddings = openai.Embedding.create(
    input=text,
    model=EMBEDDING_MODEL,
  )["data"][0]["embedding"]

  return {
    "id": raw_ad["id"],
    "project_name": project_name,
    "embeddings": embeddings,
  }


def insert_ads(collection: Collection, ads: list[dict]) -> None:
    collection.insert(ads)
    collection.flush()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a given JSON dump for ads.")
    parser.add_argument("--path", type=str, required=True, help="Path to the JSON dump file")
    parser.add_argument("--project-name", type=str, required=True, help="Project name to use")

    args = parser.parse_args()
    return {
      "path": args.path,
      "project_name": args.project_name,
    }


async def main():
  args = parse_arguments()
  path = args["path"]
  project_name = args["project_name"]

  connect_openapi()
  raw_ads = parse_dump(path)
  raw_ads = raw_ads[:10]

  tasks = [asyncio.create_task(prepare_ad(project_name, raw_ad)) for raw_ad in raw_ads]
  results = await asyncio.gather(*tasks)

  ads = []
  failed_results = []
  for result in results:
    if type(result) == dict:
      ads.append(result)
    else:
      failed_results.append(result)

  print(f"Failed results: {failed_results}")

  connect_milvus()
  collection = Collection(name=collection_name, schema=schema)
  insert_ads(collection, ads)
  disconnect_milvus()

if __name__ == "__main__":
  asyncio.run(main())
