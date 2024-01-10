import openai
import os 
import ray
from ray import serve
from ray.serve.gradio_integrations import GradioIngress
import gradio as gr
import asyncio

api_key_env = os.getenv('ANYSCALE_API_KEY')
api_base_env = "https://api.endpoints.anyscale.com/v1"
model_13b = "meta-llama/Llama-2-13b-chat-hf"
model_7b = "meta-llama/Llama-2-7b-chat-hf"
system_content = """
You are a smart assistant trying to figure out why a Ray job has failed. Given a log, Generate a valid JSON object of most relevant error. The response should ALWAYS BE A VALID JSON format and it should be parsed in its ENTIRETY.
Object should contain the following properties:

1. relevantError: This SHOULD be up to 10 words max, a verbatim of the log line that is relevant to the error. If the error has a python exception name, then ALWAYS retain that exception name in output.
2. message: Explain in details why the error might have happened.
3. suggestedFix: This should be valid terminal command or code that can be run to fix the error.
"""

sample_log = """
ray.exceptions.RayTaskError(AttributeError): [36mray::_RayTrainWorker__execute.get_next()[39m (pid=3921, ip=10.0.43.62, actor_id=8473d91401c86736a34b007703000000, repr=<ray.train._internal.worker_group.RayTrainWorker object at 0x7f25a836a220>)
  File "/home/ray/anaconda3/lib/python3.8/site-packages/ray/train/_internal/worker_group.py", line 33, in __execute
    raise skipped from exception_cause(skipped)
  File "/home/ray/anaconda3/lib/python3.8/site-packages/ray/train/_internal/utils.py", line 129, in discard_return_wrapper
    train_func(*args, **kwargs)
  File "workloads/torch_tune_serve_test.py", line 106, in training_loop
  File "/home/ray/anaconda3/lib/python3.8/site-packages/ray/train/_internal/session.py", line 728, in wrapper
    return fn(*args, **kwargs)
  File "/home/ray/anaconda3/lib/python3.8/site-packages/ray/train/_internal/session.py", line 793, in report
    _get_session().report(metrics, checkpoint=checkpoint)
  File "/home/ray/anaconda3/lib/python3.8/site-packages/ray/train/_internal/session.py", line 599, in report
    self.checkpoint(checkpoint)
  File "/home/ray/anaconda3/lib/python3.8/site-packages/ray/train/_internal/session.py", line 436, in checkpoint
    checkpoint_type, _ = checkpoint.get_internal_representation()
AttributeError: 'Checkpoint' object has no attribute 'get_internal_representation'

Trial TorchTrainer_cb091_00000 errored after 0 iterations at 2023-08-21 05:18:41. Total running time: 31s
Error file: /mnt/cluster_storage/TorchTrainer_2023-08-21_05-18-10/TorchTrainer_cb091_00000_0_lr=0.0001_2023-08-21_05-18-10/error.txt

2023-08-21 05:18:41,954	ERROR tune.py:1142 -- Trials did not complete: [TorchTrainer_cb091_00000, TorchTrainer_cb091_00001]
2023-08-21 05:18:41,964	WARNING experiment_analysis.py:917 -- Failed to read the results for 2 trials:
- /mnt/cluster_storage/TorchTrainer_2023-08-21_05-18-10/TorchTrainer_cb091_00000_0_lr=0.0001_2023-08-21_05-18-10
- /mnt/cluster_storage/TorchTrainer_2023-08-21_05-18-10/TorchTrainer_cb091_00001_1_lr=0.0010_2023-08-21_05-18-10
Retrieving best model.
2023-08-21 05:18:41,972	WARNING experiment_analysis.py:784 -- Could not find best trial. Did you pass the correct `metric` parameter?
Traceback (most recent call last):
  File "workloads/torch_tune_serve_test.py", line 281, in <module>
    best_checkpoint_path = analysis.get_best_checkpoint(
  File "/home/ray/anaconda3/lib/python3.8/site-packages/ray/tune/analysis/experiment_analysis.py", line 618, in get_best_checkpoint
    checkpoint_paths = self.get_trial_checkpoints_paths(trial, metric)
  File "/home/ray/anaconda3/lib/python3.8/site-packages/ray/tune/analysis/experiment_analysis.py", line 587, in get_trial_checkpoints_paths
    raise ValueError("trial should be a string or a Trial instance.")
ValueError: trial should be a string or a Trial instance.

Subprocess return code: 1
[INFO 2023-08-21 05:18:43,186] anyscale_job_wrapper.py: 190  Process 828 exited with return code 1.
[INFO 2023-08-21 05:18:43,186] anyscale_job_wrapper.py: 292  Finished with return code 1. Time taken: 36.497740463000014
[WARNING 2023-08-21 05:18:43,186] anyscale_job_wrapper.py: 68  Couldn't upload to cloud storage: '/tmp/release_test_out.json' does not exist.
Completed 374 Bytes/374 Bytes (3.8 KiB/s) with 1 file(s) remaining

"""

@serve.deployment
class TextGenerationModel:
    def __init__(self, model_name):
        self.model = model_name

    def __call__(self, api_base, api_key, text):
        
        try:
            response = openai.ChatCompletion.create(
                    api_base=api_base,
                    api_key=api_key,
                    model=self.model,
                    messages=[{"role": "system", "content": system_content},
                    {"role": "user", "content": text}],
                    temperature=0.01,
                    max_tokens=4000
                )
            choice = response["choices"][0]
            message = choice["message"]
            content = message["content"]
            return content
        except Exception as e:
            return e.message
        # return api_base + " \n " + api_key + " \n " + text
 
@serve.deployment
class MyGradioServer(GradioIngress):
    def __init__(self, downstream_1, downstream_2):
        self._d1 = downstream_1
        self._d2 = downstream_2
        super().__init__(lambda: gr.Interface(
            self.summarize, 
            inputs=[
                gr.Textbox(value=api_base_env, label="API URL"),
                gr.Textbox(value=api_key_env, label="API KEY"),
                gr.Textbox(value=sample_log, label="Input prompt")
                ],
            outputs=[gr.Textbox(label="Llama 7b output"), gr.Textbox(label="Llama 13b output")]
            )
        )

    async def summarize(self, api_base, api_key, text):
        refs = await asyncio.gather(self._d1.remote(api_base, api_key, text), self._d2.remote(api_base, api_key, text))
        [res1, res2] = ray.get(refs)
        return (
            f"{res1}\n\n",
            f"{res2}\n\n"
        )

app1 = TextGenerationModel.bind(model_7b)
app2 = TextGenerationModel.bind(model_13b)
app = MyGradioServer.bind(app1, app2)
