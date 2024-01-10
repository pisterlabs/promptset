import os
import logging

from evals.base import RunSpec
from evals.record import LocalRecorder
from evals.completion_fns.openai import OpenAIChatCompletionFn

from arithmetic.eval import Arithmetic

# configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO
)
logging.getLogger("openai").setLevel(logging.WARN)

# create CompletionFn
model = "gpt-3.5-turbo"
completion_fn = OpenAIChatCompletionFn(model, n_ctx=4096)

# create recorder (requires run_spec)
base_eval = "arithmetic"
split = "default"
eval_name = f"{base_eval}.{split}"
run_spec = RunSpec(
    completion_fns=[model],
    eval_name=eval_name,
    base_eval=base_eval,
    split=split,
    run_config={},
    created_by=os.getenv("USER", "")
)
record_path =  f"/tmp/evallogs/{run_spec.run_id}_{model}_{eval_name}.jsonl"
recorder = LocalRecorder(record_path, run_spec)

# create Arithmatic eval (provide paths to data)
data_dir = os.path.join("arithmetic", "data")
eval = Arithmetic(
    name=eval_name,
    train_jsonl=os.path.join(data_dir, "train.jsonl"), 
    test_jsonl=os.path.join(data_dir, "test.jsonl"),
    completion_fns=[completion_fn]
)

# run the evaluation
result = eval.run(recorder)
recorder.record_final_report(result)

# log results
logger.info(f"Run completed: {run_spec.run_id}")
for key, value in result.items():
    logger.info(f"{key}: {value}")

