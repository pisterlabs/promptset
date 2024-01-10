#!/usr/bin/env python
# coding: utf-8

# # Fine-Tune Whisper With 🤗 Transformers and Streaming Mode

# In this Colab, we present a step-by-step guide on fine-tuning Whisper with Hugging Face 🤗 Transformers and Datasets in less than X lines of code. Using streaming mode, we'll show how you can train a speech recongition model on any dataset, irrespective of size! With streaming mode, storage requirements are no longer a consideration: you can train a model on whatever dataset you want, even if it's download size exceeds your devices disk space. How can this be possible? It simply seems too good to be true! Well, rest assured it's not 😉 Carry on reading to find out more.

# ## Introduction

# Speech recognition datasets are large. A typical speech dataset consists of approximately 100 hours of audio-transcription data, requiring upwards of 130GB of storage space for download and preparation. For most ASR researchers, this is already at the upper limit of what is feasible for disk space. So what happens when we want to train on a larger dataset? The full [LibriSpeech](https://huggingface.co/datasets/librispeech_asr) dataset consists of 960 hours of audio data. Kensho's [SPGISpeech](https://huggingface.co/datasets/kensho/spgispeech) contains 5,000 hours of audio data. ML Commons [People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech) contains **30,000+** hours of audio data! Do we need to bite the bullet and buy additional storage? Or is there a way we can train on these datasets with just X GB of disk capacity?
# 
# When training machine learning systems, we rarely use the entire dataset at once. We typically _batch_ our data into smaller subsets of data, and pass these incrementally through our training pipeline. This is because we train our system on an accelerator device, such as a GPU or TPU, which has a memory limit typically around 16GB. We have to fit our model, optimiser and training data all on the same accelerator device, so we usually divide the dataset up into batches and move them from the CPU to the GPU when required. Consequently, we don't require the entire dataset to be downloaded at once - we simply need the batch of data that we pass to our model at any one go. We can leverage this principle of partial dataset download when preparing our dataset - rather than downloading the enitre dataset at the start, we can download each piece of data as and when we need it, pass it through the training pipeline, and then delete it once we are finished. In doing so, we only ever need as much disk space as each individual batch requires!
# 
# However, downloading the data on a batch-by-batch basis has a drawback. When training, we typically _shuffle_ our training data to randomly order it across the dataset. This helps the model learn **global** statistics with each training batch, rather than **local** patterns. The problem with downloading the data on a batch-by-batch basis is that we'll only ever use **consecutive** data samples in our training batches, those that appear next to each other in the training dataset. Ideally, we want to be able to perform some kind of shuffling across our dataset. Therefore, we extrapolate somewhere inbetween the batch-by-batch and full dataset approach. We download a small _subset_ of the dataset at a time: more data than in a batch, but less than the full dataset. We then shuffle the data across this subset and form batches for training. This gives an approximation to full dataset shuffling. We can set the size of this subset based on our disk space requirements. The smaller this subset, the less storage required, but the closer to a batch-by-batch approach. The larger this subset, the more storage required, but the closer to full dataset shuffling.
# 
# While the principle of subset download sounds ideal, it also seems **pretty** difficult to do. Luckily for us, 🤗 Datasets allows us to do this with minimal code changes! We'll make use of the prinicple of [_streaming_](https://huggingface.co/docs/datasets/stream). Streaming achieves exactly this: the data is downloaded progressively as we iterate over the dataset, meaning it is only downloaded as and when we need it for training. If you're familiar with training 🤗 models, the content of this notebook will be familiar, with some small extensions to support streaming mode.

# <figure>
# <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/streaming.gif" alt="Trulli" style="width:100%">
# <figcaption align = "center"><b>Figure 1:</b> Streaming mode. The dataset is divided into smaller subsets, with subsets downloaded progressively as we iterate through the dataset. </figcaption>
# </figure>

# This notebook applies to two speech related tasks: speech recognition and speech translation. Speech recognition is the task of 
# mapping from speech to text, where both are in the same language (e.g. Spanish audio -> Spanish text). In speech translation, the 
# text is a different language to the speech input (e.g. Spanish audio -> French text). Speech recognition is further divided into 
# English-only or multilingual (all other languages).
# 
# As for our model, we'll fine-tune the Whisper model released in [September 2022](https://openai.com/blog/whisper/) by the authors 
# Alec Radford et al. from OpenAI. Whisper is an encoder-decoder model pre-trained on 680k hours of labelled audio-transcription data. 
# It achieves strong performance on many speech recognition and speech translation datasets without fine-tuning. With fine-tuning, 
# we aim to improve upon these results further, with many SoTA results up for grabs!
# 
# The Whisper checkpoints come in five configurations of varying model sizes.
# The smallest four are trained on either English-only or multilingual data.
# The largest checkpoint is multilingual only. All nine of the pre-trained checkpoints 
# are available on the [Hugging Face Hub](https://huggingface.co/models?search=openai/whisper). The 
# checkpoints are summarised in the following table with links to the models on the Hub:
# 
# | Size   | Layers | Width | Heads | Parameters | English-only                                         | Multilingual                                      |
# |--------|--------|-------|-------|------------|------------------------------------------------------|---------------------------------------------------|
# | tiny   | 4      | 384   | 6     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny.)  |
# | base   | 6      | 512   | 8     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)   |
# | small  | 12     | 768   | 12    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)  |
# | medium | 24     | 1024  | 16    | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium) |
# | large  | 32     | 1280  | 20    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)  |
# 
# When fine-tuning on an English dataset for speech recognition, it is recommeneded to select one of the English-only checkpoints. For multilingual speech 
# recognition or speech translation, it is recommended to select a multilingual checkpoint.
# 
# For demonstration purposes, we'll fine-tune the multilingual version of the 
# [`"small"`](https://huggingface.co/openai/whisper-small) checkpoint with 244M params (~= 1GB). 
# As for our data, we'll train and evaluate our system on a high-resource language 
# taken from the [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)
# dataset. We'll show how we can train a model on X hours of training data using the default disk space 
# that comes with a Google Colab.

# ## Prepare Environment

# First of all, let's try to secure a decent GPU for our Colab! Unfortunately, it's becoming much harder to get access to a good GPU with the free version of Google Colab. However, with Google Colab Pro one should have no issues in being allocated a V100 or P100 GPU.
# 
# To get a GPU, click _Runtime_ -> _Change runtime type_, then change _Hardware accelerator_ from _None_ to _GPU_.

# We can verify that we've been assigned a GPU and view its specifications:

# In[ ]:


gpu_info = get_ipython().getoutput('nvidia-smi')
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)


# Next, we need to update the Unix package `ffmpeg` to version 4:

# In[ ]:


get_ipython().system('add-apt-repository -y ppa:jonathonf/ffmpeg-4')
get_ipython().system('apt update')
get_ipython().system('apt install -y ffmpeg')


# We'll employ several popular Python packages to fine-tune the Whisper model.
# We'll use `datasets` to download and prepare our training data and 
# `transformers` to load and train our Whisper model. We'll also require
# the `soundfile` package to pre-process audio files, `evaluate` and `jiwer` to
# assess the performance of our model. Finally, we'll
# use `gradio` to build a flashy demo of our fine-tuned model.

# In[ ]:


get_ipython().system('pip install datasets>=2.6.1')
get_ipython().system('pip install git+https://github.com/huggingface/transformers')
get_ipython().system('pip install librosa')
get_ipython().system('pip install evaluate>=0.30')
get_ipython().system('pip install jiwer')
get_ipython().system('pip install gradio')


# Linking the notebook to the Hugging Face Hub is straightforward - it simply requires entering your 
# Hub authentication token when prompted. Find your Hub authentication token [here](https://huggingface.co/settings/tokens):

# In[ ]:


from huggingface_hub import notebook_login

notebook_login()


# ## Load Dataset with Streaming

# This is where the magic happens! We'll first write a wrapper function around 🤗 Datasets `load_dataset` method. This function downloads the required splits using streaming mode by forcing `streaming=True` in the `load_dataset` method. Multiple splits can be combined (interleaved) by concatenating them with the "+" symbol when specifying the split name, e.g. `split=train+validation` will return a single split with the training and validation splits interleaved together. The function has the same arguments and key-word arguments as 🤗 Datasets `load_dataset` method, so we can use it in exactly the same way!

# In[1]:


from datasets import interleave_datasets, load_dataset

def load_streaming_dataset(dataset_name, dataset_config_name, split, **kwargs):
    if "+" in split:
        # load multiple splits separated by the `+` symbol *with* streaming mode
        dataset_splits = [load_dataset(dataset_name, dataset_config_name, split=split_name, streaming=True, **kwargs) for split_name in split.split("+")]
        # interleave multiple splits to form one dataset
        interleaved_dataset = interleave_datasets(dataset_splits)
        return interleaved_dataset
    else:
        # load a single split *with* streaming mode
        dataset = load_dataset(dataset_name, dataset_config_name, split=split, streaming=True, **kwargs)
        return dataset


# We'll train our system on the Spanish ("es") split of [Common Voice 11](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0). We can see how much validated data we have by heading over to the Mozilla Foundation Common Voice page and selecting "Spanish" from the drop-down menu: https://commonvoice.mozilla.org/en/datasets
# 
# This split has over 400 hours of annotated data, so we have a large abundance of training data! Since Spanish is relatively high-resource, we'll only use the `train` split for training and the `test` split for evaluation. If you're training on a low-resource language, such as Hindi which only has 12 hours of validated data, it's worth combining the `train` and `validation` splits to give a larger training set. You can achieve this by setting: `split="train+validation"` for the training split.
# 
# If you're using a gated dataset, such as Common Voice 11, ensure you have accepted the terms of use on the Hugging Face Hub: [mozilla-foundation/common_voice_11_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0). Once you have accepted the terms, you will have full access to the dataset and be able to download the data locally.

# In[5]:


from datasets import IterableDatasetDict

raw_datasets = IterableDatasetDict()

raw_datasets["train"] = load_streaming_dataset("mozilla-foundation/common_voice_11_0", "es", split="train", use_auth_token=True)  # set split="train+validation" for low-resource
raw_datasets["test"] = load_streaming_dataset("mozilla-foundation/common_voice_11_0", "es", split="test", use_auth_token=True)


# ## Prepare Processor and Data

# The ASR pipeline can be de-composed into three stages: 
# 1) A feature extractor which pre-processes the raw audio-inputs
# 2) The model which performs the sequence-to-sequence mapping 
# 3) A tokenizer which post-processes the model outputs to text format
# 
# In 🤗 Transformers, the Whisper model has an associated feature extractor and tokenizer, 
# called [WhisperFeatureExtractor](https://huggingface.co/docs/transformers/main/model_doc/whisper#transformers.WhisperFeatureExtractor)
# and [WhisperTokenizer](https://huggingface.co/docs/transformers/main/model_doc/whisper#transformers.WhisperTokenizer) 
# respectively. To make our lives simple, these two objects are wrapped under a single class, called the [WhisperProcessor](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperProcessor). We can call the WhisperProcessor to perform 
# both the audio pre-processing and the text token post-processing. In doing so, we only need to keep track of two objects during training: 
# the `processor` and the `model`.
# 
# You should set the `"language"` to your target text language. You should set the task to `"transcribe"` for speech recogntition and `"translate"` for speech translation. These arguments modify the behaviour of the tokenizer - they should be set correctly to ensure the target labels are encoded properly.

# In[ ]:


from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Spanish", task="transcribe")


# ### Prepare Data

# Let's have a look at the dataset features. Pay particular attention to the `"audio"` column - this details the sampling rate of our audio inputs:

# In[ ]:


raw_datasets["train"].features


# Since 
# our input audio is sampled at 48kHz, we need to _downsample_ it to 
# 16kHz prior to passing it to the Whisper feature extractor, 16kHz being the sampling rate expected by the Whisper model. 
# 
# We'll set the audio inputs to the correct sampling rate using dataset's 
# [`cast_column`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=cast_column#datasets.DatasetDict.cast_column)
# method. This operation does not change the audio in-place, 
# but rather signals to `datasets` to resample audio samples _on the fly_ the 
# first time that they are loaded:

# In[ ]:


from datasets import Audio

raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16000))


# Now we can write a function to prepare our data ready for the model:
# 1. We load and resample the audio data by calling `batch["audio"]`. As explained above, 🤗 Datasets performs any necessary resampling operations on the fly.
# 2. We use the feature extractor to compute the log-Mel spectrogram input features from our 1-dimensional audio array.
# 3. We encode the transcriptions to label ids through the use of the tokenizer.

# In[ ]:


def prepare_dataset(batch):
    # load and (possibly) resample audio datato 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


# We can apply the data preparation function to all of our training examples using dataset's `.map` method. We'll remove all of the columns from the raw training data, leaving just the `input_features` and `labels` defined in the `prepare_dataset` function. 
# 
# We'll treat the `"train"` and `"test"` sets differently - we want to shuffle the data in the train split, but not necessarily that of the test split. 
# 
# The size of the subset we download before shuffling is set by the variable `buffer_size`. You can increase or decrease this depending on your disk space constraints.

# In[ ]:


vectorized_datasets = IterableDatasetDict()

for split, dataset in raw_datasets.items():
    vectorized_datasets[split] = (
        dataset.map(prepare_dataset).remove_columns(list(raw_datasets[split].features.keys())).with_format("torch")
    )
    if split == "train":
        vectorized_datasets[split] = vectorized_datasets[split].shuffle(
            buffer_size=500,
            seed=0,
        )


# ## Training and Evaluation

# Now that we've prepared our data, we're ready to dive into the training pipeline. 
# The [🤗 Trainer](https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer)
# will do much of the heavy lifting for us. All we have to do is:
# 
# - Define a data collator: the data collator takes our pre-processed data and prepares PyTorch tensors ready for the model.
# 
# - Evaluation metrics: during evaluation, we want to evaluate the model using the [word error rate (WER)](https://huggingface.co/metrics/wer) metric. We need to define a `compute_metrics` function that handles this computation.
# 
# - Load a pre-trained checkpoint: we need to load a pre-trained checkpoint and configure it correctly for training.
# 
# - Define the training configuration: this will be used by the 🤗 Trainer to define the training schedule.
# 
# Once we've fine-tuned the model, we will evaluate it on the test data to verify that we have correctly trained it 
# to transcribe speech in Hindi.

# ### Define a Data Collator

# The data collator for a sequence-to-sequence speech model is unique in the sense that it 
# treats the `input_features` and `labels` independently: the  `input_features` must be 
# handled by the feature extractor and the `labels` by the tokenizer.
# 
# The `input_features` are already padded to 30s and converted to a log-Mel spectrogram 
# of fixed dimension by action of the feature extractor, so all we have to do is convert the `input_features`
# to batched PyTorch tensors. We do this using the feature extractor's `.pad` method with `return_tensors=pt`.
# 
# The `labels` on the other hand are un-padded. We first pad the sequences
# to the maximum length in the batch using the tokenizer's `.pad` method. The padding tokens 
# are then replaced by `-100` so that these tokens are **not** taken into account when 
# computing the loss. We then cut the BOS token from the start of the label sequence as we 
# append it later during training.
# 
# We can leverage the `WhisperProcessor` we defined earlier to perform both the 
# feature extractor and the tokenizer operations:

# In[ ]:


import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# Let's initialise the data collator we've just defined:

# In[ ]:


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# ### Evaluation Metrics

# We'll use the word error rate (WER) metric, the 'de-facto' metric for assessing 
# ASR systems. For more information, refer to the WER [docs](https://huggingface.co/metrics/wer). We'll load the WER metric from 🤗 Evaluate:

# In[ ]:


import evaluate

metric = evaluate.load("wer")


# We then simply have to define a function that takes our model 
# predictions and returns the WER metric. This function, called
# `compute_metrics`, first replaces `-100` with the `pad_token_id`
# in the `label_ids` (undoing the step we applied in the 
# data collator to ignore padded tokens correctly in the loss).
# It then decodes the predicted and label ids to strings. Finally,
# it computes the WER between the predictions and reference labels:

# In[ ]:


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# ### Load a Pre-Trained Checkpoint

# Now let's load the pre-trained Whisper `small` checkpoint. Again, this 
# is trivial through use of 🤗 Transformers!

# In[ ]:


from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")


# Override generation arguments - no tokens are forced as decoder outputs (see [`forced_decoder_ids`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.forced_decoder_ids)), no tokens are suppressed during generation (see [`suppress_tokens`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.suppress_tokens)):

# In[ ]:


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


# ### Define the Training Configuration

# In the final step, we define all the parameters related to training. For more detail on the training arguments, refer to the Seq2SeqTrainingArguments [docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments).

# In[ ]:


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-es",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)


# **Note**: if one does not want to upload the model checkpoints to the Hub, 
# set `push_to_hub=False`.

# We then define a custom [Callback](https://huggingface.co/docs/transformers/main_classes/callback) that is called by the 🤗 Trainer on the end of each epoch. The Callback reinitialises and reshuffles the streaming dataset at the beginning of each new epoch - this gives different shuffling across our subsets for every epoch.

# In[3]:


from transformers import TrainerCallback
from transformers.trainer_pt_utils import IterableDatasetShard
from torch.utils.data import IterableDataset

# trainer callback to reinitialise and reshuffle the streamable datasets at the beginning of each epoch
class ShuffleCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        if isinstance(train_dataloader.dataset, IterableDatasetShard):
            pass  # set_epoch() is handled by the Trainer
        elif isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)


# We can forward the training arguments to the 🤗 Trainer along with our model,
# dataset, data collator, `compute_metrics` function and custom callback:

# In[4]:


from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=vectorized_datasets["train"],
    eval_dataset=vectorized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
    callbacks=[ShuffleCallback()],
)


# In[ ]:


model.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)


# ### Training

# Training will take approximately 5-10 hours depending on your GPU or the one 
# allocated to this Google Colab. If using this Google Colab directly to 
# fine-tune a Whisper model, you should make sure that training isn't 
# interrupted due to inactivity. A simple workaround to prevent this is 
# to paste the following code into the console of this tab (_right mouse click_ 
# -> _inspect_ -> _Console tab_ -> _insert code_).

# ```javascript
# function ConnectButton(){
#     console.log("Connect pushed"); 
#     document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
# }
# setInterval(ConnectButton, 60000);
# ```

# The peak GPU memory for the given training configuration is approximately 15.8GB. 
# Depending on the GPU allocated to the Google Colab, it is possible that you will encounter a CUDA `"out-of-memory"` error when you launch training. 
# In this case, you can reduce the `per_device_train_batch_size` incrementally by factors of 2 
# and employ [`gradient_accumulation_steps`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments.gradient_accumulation_steps)
# to compensate.
# 
# To launch training, simply execute:

# In[16]:


trainer.train()


# Our best WER is 32.0% - not bad for 8h of training data! We can submit our checkpoint to the [`hf-speech-bench`](https://huggingface.co/spaces/huggingface/hf-speech-bench) on push by setting the appropriate key-word arguments (kwargs):

# In[ ]:


kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_11_0",
    "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
    "dataset_args": "config: es, split: test"
    "language": "es",
    "model_name": "Whisper Small Es - Sanchit Gandhi",  # a 'pretty' name for our model
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
}


# The training results can now be uploaded to the Hub. To do so, execute the `push_to_hub` command:

# In[ ]:


trainer.push_to_hub(**kwargs)

