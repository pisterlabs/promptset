import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import spacy
import torch
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.utils import enforce_stop_tokens
from lightning import LightningModule
from torch.nn import ParameterDict
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from tqdm import tqdm
from transformers import AutoProcessor, BatchEncoding

# A logger for this file
log = logging.getLogger(__name__)


class CLIPDescriptionsOnTheFlyLitModule(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        model: torch.nn.Module,
        processor: AutoProcessor,
        llm,
        data_path: str,
        num_classes: int,
        seed: int,
        characters: list[str] = ["2 year old", "4 year old"],
        frozen: bool = True,
        n_words: int = 45,
        impersonation_command: str = "If you were a",
        template="""{impersonation_command} {character}, how would you answer the following question in {n} words?
Q: What is an {class_name}?
A: It is""",
        stop: list[str] = ["Q:", "A:"],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.characters = characters
        self.llm = llm
        self.n_words = n_words

        # this is a work around, because configuring hydra applications with formatting strings breaks for some reason due to the curly braces
        # this way the impersonation command is configurable (as it has no curly braces) and the template is still a formatable string.
        assert "{impersonation_command}" in template
        self.impersonation_command = impersonation_command
        self.template = template.replace(
            "{impersonation_command}", impersonation_command
        )

        self.stop = stop
        self.seed = seed
        self.num_classes = num_classes
        self.data_path = data_path

        log.info("load spacy for some sentence cleaning")
        self.nlp = spacy.load("en_core_web_sm")

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["llm", "model"])

        # metrics:
        # metric objects for calculating and averaging accuracy across batches
        # Wrapping them with ParameterDict's nicely handles all the moving to devices, etc.
        self.train_accs = ParameterDict(
            {
                character: Accuracy(task="multiclass", num_classes=num_classes)
                for character in characters
            }
        )
        self.val_accs = ParameterDict(
            {
                character: Accuracy(task="multiclass", num_classes=num_classes)
                for character in characters
            }
        )
        self.test_accs = ParameterDict(
            {
                character: Accuracy(task="multiclass", num_classes=num_classes)
                for character in characters
            }
        )

        # for averaging loss across batches
        self.train_losses = ParameterDict(
            {character: MeanMetric() for character in characters}
        )
        self.val_losses = ParameterDict(
            {character: MeanMetric() for character in characters}
        )
        self.test_losses = ParameterDict(
            {character: MeanMetric() for character in characters}
        )

        # for tracking best so far validation accuracy
        self.val_acc_bests = ParameterDict(
            {character: MaxMetric() for character in characters}
        )

        # Model related stuff
        self.model = model
        self.processor = processor

        # freeze if required
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

        self.criterion = torch.nn.CrossEntropyLoss()

    def _generate_descriptions(self):
        log.info("Starting to generate the descriptions")

        # we need the dataset at least
        dataset = self.trainer.datamodule.data_train

        # use caching to avoid re-genering descriptions all the time
        # changing the order in this path will invalidate all previous paths
        cleaned_template = self.template.replace("\n", "NEWLINE")
        cached_file = Path(
            self.data_path,
            "descriptions",
            f"template={cleaned_template}",
            f"stop={' '.join(self.stop)}",
            f"llm={self.llm.model_path.replace('/','_')}",
            f"dataset={dataset.__module__}",
            f"characters={', '.join(self.characters)}",
            f"seed={int(self.seed)}",
            "generated_descriptions.json",
        )
        metadata = {
            "template": self.template,
            "stop": self.stop,
            "model_path": self.llm.model_path,
            "dataset": dataset.__module__,
            "seed": self.seed,
            "characters": list(self.characters),
        }

        if cached_file.is_file():
            try:
                with open(cached_file) as f:
                    log.info("Found cached descriptions...")
                    self.descriptions = json.load(f)
                    read_metadata = self.descriptions.pop("_metadata", None)

                    if read_metadata == metadata:
                        return
                    else:
                        log.warning("found invalid metadata, regenerating descriptions")
            except:
                pass

        log.info("Generating new descriptions")
        # check a path and look if a descriptions files exists
        self.descriptions = defaultdict(lambda: defaultdict(str))
        self.descriptions["_metadata"] = metadata

        class_names: list[str] = list(dataset.idx_to_class.values())

        # create descriptions for all characters
        for character in tqdm(self.characters):
            if character == "class_name":
                # add the class names as baseline
                descs = class_names

            else:
                # run the llm and clean the result of it to obtain descriptions
                descs = self._generate_cleaned_description(class_names, character)

            # store the descriptions
            for desc, class_name in zip(descs, class_names):
                self.descriptions[character][class_name] = desc

        # completed, save results
        # dump to JSON? how to name the file?
        # use chaching
        log.info(f"Saving descriptions to file: {cached_file}")
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cached_file, "w") as f:
            json.dump(self.descriptions, f)

    def _generate_cleaned_description(self, class_names, character):
        # get a local copy of the template
        template = self.template

        # prepare formatting data

        formatting_data = {"character": character, "n": self.n_words}
        # if the character is "none", delete the imagine part. Due to zsh this cannot be passed to ``template`` nicely
        if character.lower() == "none":
            template = self.template.replace(
                f"{self.impersonation_command} {{character}}, how", "How"
            )
            # remove the character formatting data, to keep the PromptTemplate from blowing up :/...
            formatting_data.pop("character")

        # some lightweight setup, repeating it for each character shouldn't be a problem
        input_variables = re.findall(r"\{(.+?)\}", template)

        prompt = PromptTemplate(
            input_variables=input_variables,
            template=template,
        )

        # create the prompts
        prompts = [
            prompt.format(class_name=class_name, **formatting_data)
            for class_name in class_names
        ]

        # run it through the LLM
        llm_result: list = self.llm.generate(prompts)  # type: ignore

        # save the uncleaned descriptions
        for desc, class_name in zip(llm_result, class_names):
            self.descriptions[character][f"{class_name}_uncleaned"] = desc

        # extract text from the llm result and enforece stop tokens
        descs = [enforce_stop_tokens(gen, self.stop) for gen in llm_result]

        # clean result
        log.info("Cleaning the descriptions")
        descs = [self._remove_i_am(d) for d in descs]
        descs = [
            self._remove_class_names(d, class_name)
            for d, class_name in zip(descs, class_names)
        ]
        descs = [re.sub("<[^<]+?>", "", d) for d in descs]
        return descs

    def _remove_i_am(self, text: str):
        try:
            # The :\n\n should also be a new sentence, so we convert it to a period.
            text = text.replace(":\n\n", ". ")

            doc = self.nlp(text)

            # remove sentence with "language model"
            good_sents = [
                sentence.text
                for sentence in doc.sents
                if "language model".lower() not in sentence.text.lower()
                and "I can provide you".lower() not in sentence.text.lower()
                and "I'm sorry,".lower() not in sentence.text.lower()
                and not re.match(
                    r"if i were a \d+?[- ]year[- ]old", sentence.text.lower()
                )
            ]
            return " ".join(good_sents)
        except Exception as e:
            log.warning(e)
            return text

    def _contains_class_name(self, text: str, class_name: str) -> bool:
        return class_name.lower() in text.lower()

    def _remove_class_names(self, text: str, class_name: str) -> str:
        # first some manual cleaning, which can be faster
        if self._contains_class_name(text, class_name):
            text = self._manual_cleaning(text, class_name)

        # next use the LLM for the hard cases
        if self._contains_class_name(text, class_name):
            text = self._llm_based_cleaning(text, class_name)

        return text

    def _manual_cleaning(self, text: str, class_name: str) -> str:
        for trafo in [
            lambda x: x,
            lambda x: x.lower(),
            lambda x: x.capitalize(),
        ]:
            for verb in ["is", "has", "plays", "feeds"]:
                text = text.replace(f"The {trafo(class_name)} {verb}", f"It {verb}")
                text = text.replace(f"A {trafo(class_name)} {verb}", f"It {verb}")
                text = text.replace(f"An {trafo(class_name)} {verb}", f"It {verb}")
                text = text.replace(f"{trafo(class_name)} {verb}", f"It {verb}")
            for verb in ["are", "can", "play", "feed"]:
                text = text.replace(f"{trafo(class_name)}es {verb}", f"They {verb}")
                text = text.replace(f"{trafo(class_name)}s {verb}", f"They {verb}")
                text = text.replace(f"{trafo(class_name)} {verb}", f"They {verb}")
            text = text.replace(f"It is called a {trafo(class_name)} because it", "It")
            text = text.replace(
                f"It is called the {trafo(class_name)} because it", "It"
            )
            text = text.replace(f"It is called {trafo(class_name)} because it", "It")
        return text

    def _llm_based_cleaning(self, text: str, class_name: str) -> str:
        # Now analyze individual sentences
        doc = self.nlp(text)

        template = """Replace the concept "{class_name}" from the following sentence by replacing it with indirect references / pronouns.

Sentence: The american Crow runs quickly through the forest.
Cleaned Sentence: It runs quickly through the forest.

Sentence: Warblers are small animals.
Cleaned Sentence: They are small animals.

Sentence: I like listening to nighthawks when I got to bed.
Cleaned Sentence: I like listening to them when I got to bed.

Sentence: Unfortunately, the population of sparrows has been declining.
Clean Sentence: Unfortunately, their population has been declining.

Sentence: {desc}
Cleaned Sentence:"""

        prompt = PromptTemplate(
            input_variables=["class_name", "desc"],
            template=template,
        )

        llm = HuggingFacePipeline(pipeline=self.llm)

        chain = LLMChain(llm=llm, prompt=prompt)

        sents = []
        for sent in doc.sents:
            sentence_str = str(sent)
            if self._contains_class_name(sentence_str, class_name):
                # the sentence actually contains the class
                # Run the chain only specifying the input variable.
                # run a LLM
                cleaned_sentence = chain.run(
                    class_name=class_name,
                    desc=sentence_str,
                    stop=["Sentence:", "Cleaned Sentence:", "\n\n"],
                ).strip()

                print(f"Sentence: {sentence_str}\nCleaned Sentence: {cleaned_sentence}")

                if self._contains_class_name(str(cleaned_sentence), class_name):
                    # it was not successful, just use the original
                    sents.append(sentence_str)
                else:
                    sents.append(cleaned_sentence)
            else:
                # otherwise just take the sentence as is
                sents.append(sentence_str)

        joined_sentence = " ".join(sents)
        return joined_sentence

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for character in self.characters:
            self.val_losses[character].reset()
            self.val_accs[character].reset()
            self.val_acc_bests[character].reset()

    def setup(self, stage):
        # prepare the descriptions if that has not happend
        self._generate_descriptions()

        # now delete the llm # and load the main model
        del self.llm
        # self.model = self.model()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def module_step(self, batch: dict, batch_idx: int):
        # one method to do it all
        image = batch["image"]
        label = batch["label"]

        # obtain ordered list of descriptions
        ordered_class_names = [
            self.trainer.datamodule.data_test.idx_to_class[i]
            for i in range(self.num_classes)
        ]

        # now we want to run this for each character
        return_values = {}
        for character in self.characters:
            descriptions = [
                self.descriptions[character][class_name]
                for class_name in ordered_class_names
            ]

            text = [f"A photo of {desc}" for desc in descriptions]

            # prepare all class descriptions and the image
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            # trim too long texts, this introduces an error...
            max_length = self.model.config.text_config.max_position_embeddings
            if inputs.input_ids.size(1) > max_length:
                # this should be log.warning, but produces very noisy logs, because the errors appear multiple times and contain stack traces
                print(
                    f"WARNING: trimming of too long descriptions ({inputs.input_ids.size(1)} tokens)"
                )
                inputs = BatchEncoding(
                    {key: value[:, :max_length] for key, value in inputs.items()}
                )

            # run this through the CLIP model
            outputs = self.model(**inputs)

            # this is the image-text similarity score
            logits_per_image = outputs.logits_per_image

            # we can take the softmax to get the label probabilities
            probs = logits_per_image.softmax(dim=1)
            pred_classes = probs.argmax(dim=1)

            # compute the loss and return it
            loss = self.criterion(probs, label)

            return_values[character] = {
                "loss": loss,
                "probs": probs,
                "pred_classes": pred_classes,
            }

        return return_values

    def training_step(self, batch: dict, batch_idx: int):
        label = batch["label"]
        out = self.module_step(batch, batch_idx)

        for character, results in out.items():
            loss = results["loss"]
            probs = results["probs"]

            # update and log metrics
            self.train_losses[character](loss)
            self.train_accs[character](probs, label)
            self.log(
                f"train/{character}/loss",
                self.train_losses[character],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"train/{character}/acc",
                self.train_accs[character],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return out

    def validation_step(self, batch: dict, batch_idx: int):
        label = batch["label"]
        out = self.module_step(batch, batch_idx)

        for character, results in out.items():
            loss = results["loss"]
            probs = results["probs"]
            pred_classes = results["pred_classes"]

            # update and log metrics
            self.val_losses[character](loss)
            self.val_accs[character](probs, label)
            self.log(
                f"val/{character}/loss",
                self.val_losses[character],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"val/{character}/acc",
                self.val_accs[character],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return out

    def test_step(self, batch: dict, batch_idx: int):
        label = batch["label"]
        out = self.module_step(batch, batch_idx)

        for character, results in out.items():
            loss = results["loss"]
            probs = results["probs"]
            pred_classes = results["pred_classes"]

            # update and log metrics
            self.test_losses[character](loss)
            self.test_accs[character](probs, label)
            self.log(
                f"test/{character}/loss",
                self.test_losses[character],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"test/{character}/acc",
                self.test_accs[character],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return out

    def on_validation_epoch_end(self):
        for character in self.characters:
            acc = self.val_accs[character].compute()  # get current val acc
            self.val_acc_bests[character](acc)  # update best so far val acc
            # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            self.log(
                f"val/{character}/acc_best",
                self.val_acc_bests[character].compute(),
                prog_bar=True,
            )
