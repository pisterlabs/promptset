"""Mostly adopted from OpenAI baselines: https://github.com/openai/baselines
"""
import json

import numpy as np
import pandas


def read_json(fname):
  ds = []
  with open(fname, "rt") as fh:
    for line in fh:
      ds.append(json.loads(line))
  return pandas.DataFrame(ds)


def read_csv(fname):
  return pandas.read_csv(fname, index_col=None, comment="#")


def load_csv(fname):
  with open(fname, "r") as f:
    lines = [line for line in f]
  if len(lines) < 2:
    return None
  keys = [name.strip() for name in lines[0].split(",")]
  data = np.genfromtxt(fname, delimiter=",", skip_header=1, filling_values=0.0)
  if data.ndim == 1:
    data = data.reshape(1, -1)
  assert data.ndim == 2
  assert data.shape[-1] == len(keys)
  result = {}
  for idx, key in enumerate(keys):
    result[key] = data[:, idx]
  return result
