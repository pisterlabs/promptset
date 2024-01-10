"""Provides method for retrieving opt class from string name."""
from src.emitters.opt.cma_es import CMAEvolutionStrategy
from src.emitters.opt.sep_cma_es import SeparableCMAEvolutionStrategy
from src.emitters.opt.lm_ma_es import LMMAEvolutionStrategy
from src.emitters.opt.openai_es import OpenAIEvolutionStrategy

CLASSES = {
    "cma_es": CMAEvolutionStrategy,
    "sep_cma_es": SeparableCMAEvolutionStrategy,
    "lm_ma_es": LMMAEvolutionStrategy,
    "openai_es": OpenAIEvolutionStrategy,
}


def get_class(name):
    """Retrieves opt class associated with a name."""
    if name in CLASSES:
        return CLASSES[name]
    raise ValueError(f"Unknown es '{name}'")
