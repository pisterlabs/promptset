import hydra
from omegaconf import DictConfig
import os, sys
sys.path.append(os.path.abspath(f"{os.getcwd()}"))

from src.models.prompt import Prompt
from src.models.openai import OpenAIFineTuner
from src.models.utils import env_setup

@hydra.main(config_path="../config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    env_setup()

    if cfg.model.name == "ChatGPT":
        finetuner = OpenAIFineTuner(
            Prompt(
                cfg.model.country,
                cfg.prompt.system_message_template_path,
                cfg.prompt.human_message_template_path,
            ),
            cfg.model.fine_tune.strategy,
            cfg.model.fine_tune.input_dir,
            cfg.model.fine_tune.output_dir,
            cfg.model.fine_tune.train_file,
            cfg.model.fine_tune.val_file,
        )

        finetuner.format(cfg.model.fine_tune.n)
        time.sleep(5)
        finetuner.fine_tune(cfg.model.fine_tune.n_epochs)
        # finetuner.log_fine_tune()


if __name__ == "__main__":
    main()
