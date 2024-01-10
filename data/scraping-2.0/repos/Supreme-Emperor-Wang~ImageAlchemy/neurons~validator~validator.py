import argparse
import asyncio
import copy
import os
import random
from time import sleep
from traceback import print_exception
from typing import List

import torch
from datasets import load_dataset
from neurons.constants import ENABLE_IMAGE2IMAGE, EPOCH_LENGTH, N_NEURONS
from neurons.utils import BackgroundTimer, background_loop, get_defaults
from neurons.validator.config import add_args, check_config, config
from neurons.validator.forward import run_step
from neurons.validator.reward import (
    BlacklistFilter,
    DiversityRewardModel,
    ImageRewardModel,
    NSFWRewardModel,
)
from neurons.validator.utils import (
    generate_followup_prompt_gpt,
    generate_random_prompt,
    generate_random_prompt_gpt,
    get_promptdb_backup,
    get_random_uids,
    init_wandb,
    ttl_get_block,
)
from neurons.validator.weights import set_weights
from openai import OpenAI
from transformers import pipeline

import bittensor as bt


class StableValidator:
    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    def __init__(self):
        # Init config
        self.config = StableValidator.config()
        self.check_config(self.config)
        bt.logging(config=self.config, logging_dir=self.config.alchemy.full_path)

        # Init device.
        self.device = torch.device(self.config.alchemy.device)

        # Init seed
        self.seed = random.randint(0, 1_000_000)
        bt.logging.debug(f"Seed: {self.seed}")

        # Init dataset
        bt.logging.debug("Loading dataset")
        self.dataset = iter(
            load_dataset("poloclub/diffusiondb")["train"]
            .shuffle(seed=self.seed)
            .to_iterable_dataset()
        )

        # Init prompt generation model
        bt.logging.debug(
            f"Loading prompt generation model on device: {self.config.alchemy.device}"
        )
        self.prompt_generation_pipeline = pipeline(
            "text-generation", model="succinctly/text2image-prompt-generator"
        )
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable.")
        self.openai_client = OpenAI(api_key=openai_api_key)

        # Init prompt backup db
        self.prompt_history_db = get_promptdb_backup()
        self.prompt_generation_failures = 0

        # Init subtensor
        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.debug(f"Loaded subtensor: {self.subtensor}")

        # Init wallet.
        self.wallet = bt.wallet(config=self.config)
        self.wallet.create_if_non_existent()
        if not self.config.wallet._mock:
            if not self.subtensor.is_hotkey_registered_on_subnet(
                hotkey_ss58=self.wallet.hotkey.ss58_address, netuid=self.config.netuid
            ):
                raise Exception(
                    f"Wallet not currently registered on netuid {self.config.netuid}, please first register wallet before running"
                )

        # Dendrite pool for querying the network during training.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.debug(f"Loaded dendrite pool: {self.dendrite}")

        # Init metagraph.
        self.metagraph = bt.metagraph(
            netuid=self.config.netuid, network=self.subtensor.network, sync=False
        )  # Make sure not to sync without passing subtensor
        self.metagraph.sync(subtensor=self.subtensor)  # Sync metagraph with subtensor.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.debug("Loaded metagraph")

        # Init Weights.
        self.moving_averaged_scores = torch.zeros((self.metagraph.n)).to(self.device)
        bt.logging.debug(
            f"Loaded moving_averaged_scores: {str(self.moving_averaged_scores)}"
        )

        # Each validator gets a unique identity (UID) in the network for differentiation.
        self.my_subnet_uid = self.metagraph.hotkeys.index(
            self.wallet.hotkey.ss58_address
        )
        bt.logging.info(f"Running validator on uid: {self.my_subnet_uid}")

        # Init weights
        self.weights = torch.ones_like(self.metagraph.uids, dtype=torch.float32).to(
            self.device
        )

        # Init prev_block and step
        self.prev_block = ttl_get_block(self)
        self.step = 0

        # Init reward function
        self.reward_functions = [ImageRewardModel(), DiversityRewardModel()]

        # Init reward function
        self.reward_weights = torch.tensor(
            [
                0.95,
                0.05,
            ],
            dtype=torch.float32,
        ).to(self.device)
        self.reward_weights / self.reward_weights.sum(dim=-1).unsqueeze(-1)

        # Init masking function
        self.masking_functions = [BlacklistFilter(), NSFWRewardModel()]

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Init the event loop
        self.loop = asyncio.get_event_loop()

        # Init wandb.
        init_wandb(self)
        bt.logging.debug("Loaded wandb")

        # Init blacklists and whitelists
        self.hotkey_blacklist = set()
        self.coldkey_blacklist = set()
        self.hotkey_whitelist = set()
        self.coldkey_whitelist = set()

        # Init stats
        self.stats = get_defaults(self)

        # Get vali index
        self.validator_index = self.get_validator_index()

        # Set validator request frequency
        self.request_frequency = 60
        self.query_timeout = 15

        # Start the generic background loop
        self.storage_client = None
        self.background_steps = 1
        self.background_timer = BackgroundTimer(300, background_loop, [self, True])
        self.background_timer.daemon = True
        self.background_timer.start()

    def run(self):
        # Main Validation Loop
        bt.logging.info("Starting validator loop.")
        self.step = 0
        while True:
            try:
                # Reduce calls to miner to be approximately 1 per 5 minutes
                block_diff = ttl_get_block(self) - self.prev_block
                while block_diff < 1:
                    bt.logging.info(
                        f"Waiting for {self.request_frequency} seconds before querying miners again..."
                    )
                    sleep(self.request_frequency)

                bt.logging.trace(f"Block diff: {block_diff}")

                # Get a random number of uids

                uids = get_random_uids(self, self.dendrite, k=N_NEURONS)

                uids = uids.to(self.device)

                axons = [self.metagraph.axons[uid] for uid in uids]

                # Generate prompt + followup_prompt
                prompt = generate_random_prompt_gpt(self)
                followup_prompt = generate_followup_prompt_gpt(self, prompt)
                if (prompt is None) or (followup_prompt is None):
                    if (self.prompt_generation_failures != 0) and (
                        (self.prompt_generation_failures / len(self.prompt_history_db))
                        > 0.2
                    ):
                        self.prompt_history_db = get_promptdb_backup(
                            self.prompt_history_db
                        )
                    prompt, followup_prompt = random.choice(self.prompt_history_db)
                    self.prompt_history_db.remove((prompt, followup_prompt))
                    self.prompt_generation_failures += 1

                # Text to Image Run
                t2i_event = run_step(
                    self, prompt, axons, uids, task_type="text_to_image"
                )
                if ENABLE_IMAGE2IMAGE:
                    # Image to Image Run
                    followup_image = [image for image in t2i_event["images"]][
                        torch.tensor(t2i_event["rewards"]).argmax()
                    ]
                    if (
                        (followup_prompt is not None)
                        and (followup_image is not None)
                        and (followup_image != [])
                    ):
                        _ = run_step(
                            self,
                            followup_prompt,
                            axons,
                            uids,
                            "image_to_image",
                            followup_image,
                        )
                # Re-sync with the network. Updates the metagraph.
                self.sync()

                # End the current step and prepare for the next iteration.
                self.step += 1

            # If we encounter an unexpected error, log it for debugging.
            except Exception as err:
                bt.logging.error("Error in training loop", str(err))
                bt.logging.debug(print_exception(type(err), err, err.__traceback__))

            # If the user interrupts the program, gracefully exit.
            except KeyboardInterrupt:
                bt.logging.success("Keyboard interrupt detected. Exiting validator.")
                exit()

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()

        if self.should_sync_metagraph():
            self.resync_metagraph()

        if self.should_set_weights():
            set_weights(self)
            self.prev_block = ttl_get_block(self)

    def get_validator_index(self):
        """
        Retrieve the given miner's index in the metagraph.
        """
        index = None
        try:
            index = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        except ValueError:
            pass
        return index

    def get_validator_info(self):
        return {
            "block": self.metagraph.block.item(),
            "stake": self.metagraph.S[self.validator_index],
            "rank": self.metagraph.R[self.validator_index],
            "vtrust": self.metagraph.T[self.validator_index],
            "dividends": self.metagraph.C[self.validator_index],
            "emissions": self.metagraph.E[self.validator_index],
        }

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = torch.zeros((self.metagraph.n)).to(self.device)
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def check_registered(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey before trying again"
            )
            exit()

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return (
            ttl_get_block(self) - self.metagraph.last_update[self.uid]
        ) > EPOCH_LENGTH

    def should_set_weights(self) -> bool:
        # Check if enough epoch blocks have elapsed since the last epoch.
        return (ttl_get_block(self) % self.prev_block) >= EPOCH_LENGTH
