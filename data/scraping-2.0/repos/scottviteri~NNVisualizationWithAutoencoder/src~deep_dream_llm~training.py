"""
This contains code to train the various components.

TODO: Move this outside of src/ and instead import this as a python package with pip install.
"""

import torch
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import openai
from tqdm.auto import tqdm
from accelerate import Accelerator
import numpy as np
import random
import os
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import TrainingConfig
from autoencoder import TAE, LinearAutoEncoder, Gpt2AutoencoderBoth

from utils import (
    unembed_and_decode,
    generate_sentence,
    generate_sentence_batched,
    update_plot,
    print_results,
)

class DeepDreamLLMTrainer:
    def __init__(self, config: TrainingConfig):
        """
        A class for training an autoencoder and for optimizing a sentence in the latent
        space of that autencoder in order to activate a neuron.

        Args:
            config (TrainingConfig): a configuration object containing various training parameters.

        """
        self.__dict__.update(vars(config))

        if self.use_openai:
            print("Please input your OpenAI API key in the terminal below:")
            openai.api_key = input()

        # Load the model's parameters from a checkpoint if provided
        if self.autoencoder is None:
            # Default autoencoder
            full_name = None
            model_names = ['LinearAutoEncoder', 'Gpt2AutoencoderBoth', 'TAE']
            if self.autoencoder_name not in model_names:
                full_name = self.autoencoder_name
                self.autoencoder_name = self.autoencoder_name.split('_')[0]
            if self.autoencoder_name == "LinearAutoEncoder":
                self.autoencoder = LinearAutoEncoder("distilgpt2", latent_dim=self.latent_dim)
            elif self.autoencoder_name == "Gpt2AutoencoderBoth":
                self.autoencoder = Gpt2AutoencoderBoth("distilgpt2", latent_dim=self.latent_dim)
            elif self.autoencoder_name == "TAE":
                self.autoencoder = TAE("distilgpt2", latent_dim=self.latent_dim)
            else:
                raise NotImplementedError(f"Autoencoder {config.autoencoder_name} not implemented")
            if full_name:
                if self.load_path is None:
                   self.load_path =  "/content/NNVisualizationWithAutoencoder/Checkpoints"
                loaded = torch.load(os.path.join(self.load_path, full_name))
                self.autoencoder.load_state_dict(loaded)

        accelerator = Accelerator()  # TODO actually use this other than just preparing stuff
        if self.model is None: self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.autoencoder.parameters(), lr=self.learning_rate)
        self.model, self.optimizer, self.autoencoder = accelerator.prepare(self.model, self.optimizer, self.autoencoder)
        self.device = accelerator.device
        #self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        #self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
        #    lambda epoch: 10 * epoch / 1001 if epoch / 1001 < 0.1 else 1 - epoch / 1001)
        #self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        #self.lr_scheduler = None
        #self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 1000, T_mult=2)
        self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer)
        #self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.001)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2", use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        #print("Testing autoencoder shapes")
        #self.test_autoencoder_shapes()
        #print("Autoencoder shapes test passed")

    def get_embeddings(self, input_ids):
        return self.model.transformer.wte(input_ids)

    def encode_sentence(self, sentence):
        assert isinstance(sentence, str), "sentence must be a string"
        return self.tokenizer.encode(
            sentence,
            return_tensors="pt",
        ).to(self.device)

    def embeddings_to_tokens(self, embeddings):
        """
        Args:
            embeddings (torch.tensor): the embeddings to unembed, of shape (batch_size, sequence_length, embedding_size)
        Returns:
            torch.tensor: the tokens, of shape (batch_size, sequence_length)
        """
        with torch.no_grad():
            pretrained_embeddings = self.model.transformer.wte.weight
            dot_product = torch.matmul(embeddings, pretrained_embeddings.t())
            _, tokens = torch.max(dot_product, dim=-1)
        return tokens

    # 3 kinds of loss: loss, openai_distance, and reencode_loss
    def calc_direct_loss(self, original, reconstructed):
        return (1 - torch.nn.functional.cosine_similarity(original, reconstructed, dim=-1)).sum()

    def model_embed_loss(self, original, reconstructed, attention_mask=None, layer_idx=5):
        """
        Calculate the mean squared error (MSE) between the average token embeddings
        generated by the third transformer block of a GPT-2 model, given original and
        reconstructed input sequences.

        Args:
            original (torch.Tensor): Original input sequences.
                                    Shape: (batch_size, seq_len, embedding_dim)
            reconstructed (torch.Tensor): Reconstructed input sequences.
                                          Shape: (batch_size, seq_len, embedding_dim)

        Returns:
            torch.Tensor: A scalar tensor representing the mean squared error.
        """
        # Create a list to store the output of the forward hook
        handle_output = [None]

        # Define the forward hook function
        def hook(module, input, output):
            # Save the output of the third transformer block
            handle_output[0] = output

        # Register the forward hook on the third transformer block
        handle = self.model.transformer.h[layer_idx].register_forward_hook(hook)

        # Run the original and reconstructed inputs through the model
        self.model(inputs_embeds=original, attention_mask = attention_mask)
        orig_vecs = handle_output[0][0].clone() #[batch_size, seq_len, embedding_dim]
        self.model(inputs_embeds=reconstructed, attention_mask = attention_mask)
        recon_vecs = handle_output[0][0].clone()

        # Remove the hook
        handle.remove()

        # Compute the mean of the token embeddings for both original and reconstructed inputs
        orig_vecs_token_ave = orig_vecs.mean(dim=1) #[batch_size, embedding_dim]
        recon_vecs_token_ave = recon_vecs.mean(dim=1) #[batch_size, embedding_dim]

        # Compute the MSE between the averaged embeddings for each item in the batch
        mse_per_batch = ((orig_vecs_token_ave - recon_vecs_token_ave)**2).mean(dim=-1) #[batch_size]
        cos_per_batch = 1 - torch.nn.functional.cosine_similarity(orig_vecs_token_ave, recon_vecs_token_ave, dim=1) #[batch_size]

        # Return the mean of these MSE values, effectively averaging the loss across the batch
        return cos_per_batch.mean()

    def model_embed_loss_all(self, original, reconstructed, attention_mask=None):
        """
        Calculate the mean squared error (MSE) between the average token embeddings
        generated by all transformer blocks of a GPT-2 model, given original and
        reconstructed input sequences.

        Args:
            original (torch.Tensor): Original input sequences.
                                    Shape: (batch_size, seq_len, embedding_dim)
            reconstructed (torch.Tensor): Reconstructed input sequences.
                                          Shape: (batch_size, seq_len, embedding_dim)

        Returns:
            torch.Tensor: A scalar tensor representing the mean squared error.
        """
        # Create a list to store the output of the forward hooks
        handle_outputs = [[] for _ in range(len(self.model.transformer.h))]

        # Define a function to create the hook
        def create_hook(layer_idx):
            def hook(module, input, output):
                handle_outputs[layer_idx].append(output)
            return hook

        # Register the forward hook on each transformer block
        handles = []
        for layer_idx in range(len(self.model.transformer.h)):
            hook = create_hook(layer_idx)
            handle = self.model.transformer.h[layer_idx].register_forward_hook(hook)
            handles.append(handle)

        # Run the original and reconstructed inputs through the model
        self.model(inputs_embeds=original, attention_mask=attention_mask)
        orig_vecs = [output[0][0].clone() for output in handle_outputs]  # list of tensors of shape [batch_size, seq_len, embedding_dim]
        self.model(inputs_embeds=reconstructed, attention_mask=attention_mask)
        recon_vecs = [output[0][0].clone() for output in handle_outputs]  # list of tensors of shape [batch_size, seq_len, embedding_dim]

        # Remove the hooks
        for handle in handles:
            handle.remove()

        # Compute the mean of the token embeddings for both original and reconstructed inputs
        orig_vecs_token_ave = torch.mean(torch.stack(orig_vecs), dim=0).mean(dim=1)  # [batch_size, embedding_dim]
        recon_vecs_token_ave = torch.mean(torch.stack(recon_vecs), dim=0).mean(dim=1)  # [batch_size, embedding_dim]

        # Compute the MSE between the averaged embeddings for each item in the batch
        mse_per_batch = ((orig_vecs_token_ave - recon_vecs_token_ave) ** 2).mean(dim=-1)  # [batch_size]
        cos_per_batch = 1 - torch.nn.functional.cosine_similarity(orig_vecs_token_ave, recon_vecs_token_ave, dim=1)  # [batch_size]

        # Return the mean of these MSE values, effectively averaging the loss across the batch
        return cos_per_batch.mean()

    def calc_openai_loss(self, sentence1, sentence2):
        # this loss is distinguished by using openai embeddings to measure similarity
        # this one should be the most dissimilar in nature
        response1 = openai.Embedding.create(
            input=sentence1, model="text-embedding-ada-002"
        )
        response2 = openai.Embedding.create(
            input=sentence2, model="text-embedding-ada-002"
        )
        embedding1 = np.array(response1["data"][0]["embedding"])
        embedding2 = np.array(response2["data"][0]["embedding"])
        distance = 1 - cosine_similarity(
            embedding1.reshape(1, -1), embedding2.reshape(1, -1)
        )
        return distance.item()

    def calc_reencode_loss(self, input_ids_1, input_ids_2):
        # this loss is distinguished from original loss by decoding, re-encoding and taking embeddings distance
        embeddings_1, embeddings_2 = self.get_embeddings(
            input_ids_1
        ), self.get_embeddings(input_ids_2)
        # truncate the longer embedding to be the size of the shorter one
        if embeddings_1.shape[1] > embeddings_2.shape[1]:
            embeddings_1 = embeddings_1[:, : embeddings_2.shape[1], :]
        elif embeddings_2.shape[1] > embeddings_1.shape[1]:
            embeddings_2 = embeddings_2[:, : embeddings_1.shape[1], :]
        return self.calc_direct_loss(embeddings_1, embeddings_2).item()

    def train_autoencoder(self, num_epochs, print_every, save_path=None, num_sentences=None):
        """
        Args:
            num_epochs (int): the number of epochs to train for
            print_every (int): the number of epochs to print the loss for
            save_path (Optional[str]): the path to save the model to
            num_sentences (Optional[int]): the number of sentences to generate at a time
        """
        if save_path is None:
            save_path = f"/content/NNVisualizationWithAutoencoder/Checkpoints/{self.autoencoder_name}_{num_epochs}_{print_every}.pt"
        losses, direct_losses, openai_losses, reencode_losses = [], [], [], []
        sentences, reconstructed_sentences = np.array([]), []

        #pbar = tqdm(range(num_epochs))
        pbar = range(num_epochs)
        # Training loop
        for epoch in pbar:
            # If there aren't enough sentences left, generate new ones
            if sentences.size < self.batch_size:
                print("Ran out of sentences, generating another batch")
                nsent = num_sentences if num_sentences else 1000
                new_sentences = generate_sentence_batched(
                    self.model, self.tokenizer, n=nsent
                )
                sentences = np.append(sentences, new_sentences)

            # Extract a batch of sentences and remove them from the array
            input_sentences = sentences[:self.batch_size]
            sentences = sentences[self.batch_size:]

            encoding = self.tokenizer(input_sentences.tolist(), padding=True, truncation=True, return_tensors="pt")
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            original_embeddings = self.get_embeddings(input_ids)
            reconstructed_embeddings = self.autoencoder(original_embeddings, attention_mask.T==0)
            direct_loss = self.calc_direct_loss(original_embeddings, reconstructed_embeddings)
            # not passing attention mask to GPT2
            model_embed_loss = self.model_embed_loss(original_embeddings, reconstructed_embeddings, attention_mask)
            loss = direct_loss*self.lam + model_embed_loss*(1.0-self.lam)
            loss.backward()
            #print(self.autoencoder.projection_1.weight.grad)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            losses.append(loss.item())

            if epoch % print_every == 0:
                # Record the loss value for plotting
                reconstructed_tokens = self.embeddings_to_tokens(reconstructed_embeddings)
                reconstructed_sentence = unembed_and_decode(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    embeds_input=reconstructed_embeddings,
                )[0]
                input_sentence = input_sentences[0]
                reconstructed_sentences.append(reconstructed_sentence)
                direct_loss = self.calc_direct_loss(original_embeddings, reconstructed_embeddings).item()
                direct_losses.append(direct_loss)
                reencode_loss = None
                if self.use_reencode:
                    reencode_loss = self.calc_reencode_loss(
                        input_ids[0].reshape(1, -1), reconstructed_tokens[0].reshape(1, -1)
                    )
                    reencode_losses.append(reencode_loss)
                openai_loss = None
                if self.use_openai:
                    openai_loss = self.calc_openai_loss(
                        input_sentence, reconstructed_sentence
                    )
                    openai_losses.append(openai_loss)
                if self.is_notebook:
                    from IPython.display import clear_output
                    clear_output(wait=True)
                    update_plot(losses, direct_losses, openai_losses, reencode_losses, print_every)
                print_results(
                    epoch,
                    input_sentence,
                    reconstructed_sentence,
                    loss.item(),
                    direct_loss, 
                    openai_loss,
                    reencode_loss,
                    num_epochs,
                    self.lr_scheduler.state_dict()['_last_lr'][0]
                )
                if save_path: torch.save(self.autoencoder.state_dict(), save_path)
        return losses, openai_losses, reencode_losses, sentences, reconstructed_sentences

    def neuron_loss_fn(activation):
        """
        Default loss function
        """
        return -torch.sigmoid(activation)

    def optimize_for_neuron_whole_input(
        self,
        neuron_index=0,
        layer_num=1,
        mlp_or_attention="mlp",
        num_tokens=50,
        num_iterations=200,
        loss_fn=neuron_loss_fn,
        learning_rate=0.1,
        seed=42,
        verbose=False
    ):
        """
        Args:
            neuron_index (int): the index of the neuron to optimize for
            layer_num (int): the layer number to optimize for
            mlp_or_attention (str): 'mlp' or 'attention'
            num_tokens (int): the number of tokens to in the sentence that we optimize over
            num_iterations (int): the number of iterations to run the optimization for
            loss_fn (function): the loss function to use.
            learning_rate (float): the learning rate to use for the optimizer
            seed (int): the seed to use for reproducibility
        Returns:
            losses (list): the list of losses of the model
            log_dict (dict): has keys below
                original_sentence (str): the original generated sentence
                original_sentence_reconstructed (str) : the original sentence after reconstructing it
                reconstructed_sentences (list): Reconstructed sentences during training every 1/30th of the way through
                activations (list): Average activations of the neuron every 1/30th of the way through
        """
        log_dict = {}

        # Set the seed for reproducibility
        sentence = generate_sentence(
            self.model, self.tokenizer, max_length=num_tokens, seed=seed
        )
        if verbose: tqdm.write("Original sentence is:")
        if verbose: tqdm.write(sentence)
        log_dict["original_sentence"] = sentence

        input_ids = self.encode_sentence(sentence)
        original_embeddings = self.get_embeddings(input_ids)
        latent = self.autoencoder.encode(original_embeddings, attention_mask=None) # batch size 1, no mask needed
        latent_vectors = latent.detach().clone().to(self.device)
        latent_vectors.requires_grad = True

        if verbose: tqdm.write("original reconstructed sentence is ")
        with torch.no_grad():
            og_reconstructed_sentence = unembed_and_decode(
                self.model, self.tokenizer, self.autoencoder.decode(latent_vectors, attention_mask=None)
            )
            log_dict["original_sentence_reconstructed"] = og_reconstructed_sentence
        # Create an optimizer for the latent vectors
        optimizer = AdamW(
            [latent_vectors], lr=learning_rate
        )  # You may need to adjust the learning rate

        if "mlp" in mlp_or_attention:
            layer = self.model.transformer.h[layer_num].mlp.c_fc
        elif "attention" in mlp_or_attention:
            layer = self.model.transformer.h[layer_num].attn.c_attn
        else:
            raise NotImplementedError("Haven't implemented attention block yet")

        activation_saved = [torch.tensor(0.0, device=self.device)]

        def hook(model, input, output):
            # The output is a tensor. We're getting the average activation of the neuron across all tokens.
            activation = output[0, :, neuron_index].mean()
            activation_saved[0] = activation

        handle = layer.register_forward_hook(hook)

        losses, log_dict["reconstructed_sentences"], log_dict["activations"] = [], [], []
        if verbose:
            pbar = tqdm(range(num_iterations), position=0, leave=True)
        else:
            pbar = range(num_iterations)
        for i in pbar:
            # Construct input for the self.model using the embeddings directly
            embeddings = self.autoencoder.decode(latent_vectors, attention_mask=None)
            _ = self.model(
                inputs_embeds=embeddings
            )  # the hook means outputs are saved to activation_saved
            # We want to maximize activation, which is equivalent to minimizing negative activation
            loss = loss_fn(activation_saved[0])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i % (num_iterations // 30) == 0:
                if verbose: tqdm.write(f"Loss at step {i}: {loss.item()}\n", end="")
                reconstructed_sentence = unembed_and_decode(
                    self.model, self.tokenizer, embeddings
                )[0]
                if verbose: tqdm.write(reconstructed_sentence, end="")
                log_dict["reconstructed_sentences"].append(reconstructed_sentence)
                log_dict["activations"].append(activation_saved[0].item())
            optimizer.zero_grad()

        handle.remove()  # Don't forget to remove the hook!
        return losses, log_dict

    def test_autoencoder_shapes(self):
        """
        1. Checks that the latent vector shape is the same as latent_dim
        2. Checks that the decoded shape is the same as the starting shape

        3. Unembed and decode has different number of tokens potentially :(
        """

        # 1
        sentence = generate_sentence(
            self.model, self.tokenizer, max_length=10
        )
        input_ids = self.encode_sentence(sentence)
        original_embeddings = self.get_embeddings(input_ids)
        latent = self.autoencoder.encode(original_embeddings, attention_mask=None)
        assert latent.shape[2] == self.autoencoder.latent_dim, (
            f"latent dim {latent.shape[2]} does not match autoencoder latent dim {self.autoencoder.latent_dim}"
        )

        # 2
        reconstructed_embeddings = self.autoencoder.decode(latent, attention_mask=None)
        assert reconstructed_embeddings.shape == original_embeddings.shape, (
            f"reconstructed_embeddings shape {reconstructed_embeddings.shape} does not match original_embeddings shape {original_embeddings.shape}"
        )

        return True
