import logging

import traceback
from typing import Optional
import json

from huggingface_hub.inference_api import InferenceApi
from langchain import HuggingFaceHub, OpenAI
from langchain.chains.base import Chain as BaseChain
from lib.model.chain_revision import ChainRevision, find_ancestor_ids
from lib.model.chain import Chain
from lib.model.chain_spec import ChainSpec
from lib.model.result import Result
from lib.model.lang_chain_context import LangChainContext
from lib.db import chain_revision_repository, chain_repository, result_repository
from bson import ObjectId
import dotenv
import os
import pinecone
from datetime import datetime

logging.basicConfig(level=logging.INFO)

dotenv.load_dotenv()


def initialize_services():
  if os.getenv("PINECONE_API_KEY") is not None:
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))


def save_revision(chain_name, revision) -> str:
  chain = chain_repository.find_one_by({"name": chain_name})

  if chain is not None:
    revision.parent = chain.revision
    chain.revision = revision.id
    chain_revision_repository.save(revision)

    chain.revision = revision.id
    chain_repository.save(chain)
  else:
    chain_revision_repository.save(revision)

    chain = Chain(name=chain_name, revision=revision.id)
    chain_repository.save(chain)

  return revision.id


def load_by_chain_name(chain_name):
  chain = chain_repository.find_one_by({"name": chain_name})
  return chain_revision_repository.find_one_by_id(chain.revision)


def load_by_id(revision_id):
  return chain_revision_repository.find_one_by_id(revision_id)


def history_by_chain_name(chain_name):
  """return the ids of all revisions of the chain.
  """
  chain = chain_repository.find_one_by({"name": chain_name})
  revision = chain_revision_repository.find_one_by_id(chain.revision)
  return [revision.id] + find_ancestor_ids(revision.id, chain_revision_repository)


def save_patch(chain_name, patch) -> str:
  chain = chain_repository.find_one_by({"name": chain_name})
  revision = chain_revision_repository.find_one_by_id(chain.revision)
  new_chain = revision.chain.copy_replace(lambda spec: patch if spec.chain_id == patch.chain_id else spec)

  try:
    ctx = LangChainContext(llms=revision.llms)
    revision.chain.to_lang_chain(ctx)
  except Exception as e:
    raise ValueError("Could not realize patched chain:", e)

  new_revision = revision.copy()
  new_revision.id = None
  new_revision.chain = new_chain
  new_revision.parent = revision.id

  chain_revision_repository.save(new_revision)
  chain.revision = new_revision.id
  chain_repository.save(chain)

  return revision.id


def branch(chain_name, branch_name):
  """Branch a chain revision.
  This will create a new chain that points to the same revision
  as the provided chain. The new chain may be then revised independently.
  """
  chain = chain_repository.find_one_by({"name": chain_name})

  new_chain = Chain(name=branch_name, revision=chain.revision)
  chain_repository.save(new_chain)


def reset(chain_name, revision):
  """Reset a chain to a another revision.
  This will update the chain to point to the given revision.
  """
  new_revision = chain_revision_repository.find_one_by({"id": ObjectId(revision)})
  if new_revision is None:
    raise ValueError(f"Revision {revision} not found")

  chain = chain_repository.find_one_by({"name": chain_name})
  chain.revision = new_revision.id
  chain_repository.save(chain)


def list_chains():
  """List all chains."""
  return {chain_name.name: str(chain_name.revision) for chain_name in chain_repository.find_by({})}


def compute_chain_IO(chain_spec: ChainSpec, chain: BaseChain, ctx: LangChainContext):
  def compute_and_update(chain_spec: ChainSpec, chain: BaseChain, ctx: LangChainContext, inputs, outputs):
    inner_inputs, inner_outputs = compute_chain_IO(chain_spec, chain, ctx)
    inputs = {**inputs, **inner_inputs}
    outputs = {**outputs, **inner_outputs}
    return inputs, outputs

  chain_id = chain_spec.chain_id
  inputs, outputs = {}, {}

  inputs[chain_id] = chain.input_keys
  outputs[chain_id] = chain.output_keys

  if chain_spec.chain_type == "sequential_chain_spec":
    for sub_chain_spec, sub_chain in zip(chain_spec.chains, chain.chains):
      inputs, outputs = compute_and_update(sub_chain_spec, sub_chain.chain, ctx, inputs, outputs)

  if chain_spec.chain_type == "case_chain_spec":
    for case_spec, case in zip(chain_spec.cases.values(), chain.subchains.values()):
      inputs, outputs = compute_and_update(case_spec, case.chain, ctx, inputs, outputs)

    inputs, outputs = compute_and_update(chain_spec.default_case, chain.default_chain.chain, ctx, inputs, outputs)

  return inputs, outputs

def run_once(chain_name, input, record):
  """Run a chain revision.
  The revision is read from the database and fed input from stdin or the given file.
  The results are ouput as json to stdout.
  """
  chain = chain_repository.find_one_by({"name": chain_name})
  revision = chain_revision_repository.find_one_by_id(chain.revision)

  filledLLMs = {key: llm.to_llm() for key, llm in revision.llms.items()}

  ctx = LangChainContext(llms=filledLLMs, recording=True)
  lang_chain = revision.chain.to_lang_chain(ctx)
  output = lang_chain._call(input)

  if record:
    vars = ctx.get_IO()

    inputs, outputs = compute_chain_IO(revision.chain, lang_chain.chain, ctx)

    result = Result(revisionID=revision.id, inputs=inputs, outputs=outputs, io_mapping=vars, recorded=datetime.now())
    result_repository.save(result)

  return output

def results(revision_id: str, ancestors: bool):
  """Find results for a prompt for one or more chain revisions.
  The revision must be specified.
  If the ancestors flag is set, results for all ancestors of the specified revision are included.
  """
  revision_ids = [ObjectId(revision_id)]
  # if ancestors are requested, find all ancestors of the specified revision
  if ancestors:
    ancestors_id = find_ancestor_ids(revision_id, chain_revision_repository)
    revision_ids += [ObjectId(i) for i in ancestors_id]
  return result_repository.find_by({"revisionID": {"$in": revision_ids}})


def load_results_by_chain_name(chain_name: str, ancestors: bool):
  """Find results for a prompt for one chain revision.
  The chain name must be specified.
  """
  chain = chain_repository.find_one_by({"name": chain_name})
  revision = chain_revision_repository.find_one_by_id(chain.revision)
  return results(revision.id, ancestors)


def export_chain(chain_name: str) -> str:
  export_ids = history_by_chain_name(chain_name)

  chain_revisions = [chain_revision_repository.find_one_by_id(id) for id in export_ids]
  if not all(chain_revisions):
    missing_id = next((id for id, revision in zip(export_ids, chain_revisions) if not revision), None)
    raise Exception("Could not find revision with id: " + missing_id)

  return chain_revisions[::-1]


def import_chain(chain_name, chain_details):
  root_revision = None
  for revision in chain_details:
    try:
      if not revision.parent:
        root_revision = revision
      chain_revision_repository.save(revision)
    except Exception as e:
      traceback.print_exc()

  leaf_revision = find_leaf_revision(chain_details, root_revision)

  chain = Chain(name=chain_name, revision=leaf_revision)
  chain_repository.save(chain)


def find_leaf_revision(revisions, current_revision):
  if (current_revision is None):
    return current_revision.id
  id = current_revision.id
  for revision in revisions:
    if revision.parent == id:
      return find_leaf_revision(revisions, revision)
  return id
