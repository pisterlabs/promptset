from langchain.llms.base import BaseLanguageModel
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.evaluation.embedding_distance.base import (
    EmbeddingDistance,
)
from prompt_breeder.prompts.string import (
    StringTaskPrompt,
    StringMutationPrompt,
    StringThinkingStyle,
    StringProblemDescription,
)
from prompt_breeder.provider.json_file import RandomJsonListLoad
from prompt_breeder.evolution.fitness import BestMemberFitness
from prompt_breeder.evolution.base import EvolutionExecutor
from prompt_breeder.evolution.binary_tournament import BinaryEvolution
from prompt_breeder.evolution.initialization.base import (
    PositivePopulationInitialization,
)
from prompt_breeder.evolution.initialization.zero_order_random import (
    ZeroOrderInitialization,
)
from prompt_breeder.data import FP_BASE_MUTATION_PROMPTS, FP_BASE_THINKING_STYLES
from prompt_breeder.mutators.zero_order_prompt_generation import (
    ZeroOrderPromptGeneration,
)
from prompt_breeder.mutators.first_order_prompt_generation import (
    FirstOrderPromptGeneration,
)
from prompt_breeder.mutators.estimation_of_distribution_mutation import (
    EstimationOfDistributionMutation,
)
from prompt_breeder.mutators.eda_rank_and_index_mutation import (
    EdaRankAndIndexMutation,
)
from prompt_breeder.mutators.lineage_based_mutation import (
    LineageBasedMutation,
)
from prompt_breeder.mutators.zero_order_hypermutation import (
    ZeroOrderHypermutation,
)
from prompt_breeder.mutators.first_order_hypermutation import (
    FirstOrderHypermutation,
)
from prompt_breeder.mutators.crossover import (
    PromptCrossover,
)
from prompt_breeder.mutators.elite import (
    AddElite,
)
from prompt_breeder.mutators.reinit import ReplaceWithInit
from prompt_breeder.evolution.callbacks import (
    IncrementAge,
    SavePopulation,
    TaskPromptSummary,
    UnitFitnessSummary,
)

from experiments.gsm8k.fitness import create_gsm8k_fitness


def str_task_prompt_factory(x):
    return StringTaskPrompt(text=x)


def str_mutation_prompt_factory(x):
    return StringMutationPrompt(text=x)


def str_thinkingstype_prompt_factory(x):
    return StringThinkingStyle(text=x)


def str_problem_desc_prompt_factory(x):
    return StringProblemDescription(text=x)


def create_experiment(
    cached_llm: BaseLanguageModel,
    llm: BaseLanguageModel,
    embed_model: CacheBackedEmbeddings,
    n_members_per_unit: int = 3,
    n_units: int = 20,
    ed_threshold: float = 0.05,
    crossover_prob=0.1,
    num_predict: int = 100,
    samples: int = 50,
    fp_population: str = "./population.json",
    fp_distribution: str = "./distribution_output.csv",
    fp_detailed: str = "./detailed_output.csv",
):
    fitness_scorer = create_gsm8k_fitness(
        cached_llm,
        "train",
        samples,
        llm_kwargs={"num_predict": num_predict},
    )
    val_fitness_scorer = create_gsm8k_fitness(
        cached_llm,
        "test",
        samples,
        llm_kwargs={"num_predict": num_predict},
    )
    multiple_scorer = BestMemberFitness(scorer=fitness_scorer)
    val_multiple_scorer = BestMemberFitness(scorer=val_fitness_scorer)

    cb0 = IncrementAge()
    cb1 = SavePopulation(fp=fp_population)
    cb2 = UnitFitnessSummary(
        fitness_scorer=multiple_scorer,
        val_fitness_scorer=val_multiple_scorer,
        fp=fp_distribution,
    )
    cb3 = TaskPromptSummary(
        fitness_scorer=fitness_scorer,
        val_fitness_scorer=val_fitness_scorer,
        fp=fp_detailed,
    )

    thinking_style_provider = RandomJsonListLoad(
        factory=str_task_prompt_factory, repeating=True
    ).load(fp=str(FP_BASE_THINKING_STYLES))
    mutation_prompt_provider = RandomJsonListLoad(
        factory=str_mutation_prompt_factory, repeating=True
    ).load(fp=str(FP_BASE_MUTATION_PROMPTS))

    # Diresct Mutators
    mutator_zero_order_prompt_gen = ZeroOrderPromptGeneration.from_llm(
        llm=llm,
        task_prompt_factory=str_task_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        llm_kwargs={"num_predict": num_predict},
        # verbose=1,
        # callbacks=[handler]
    )
    mutator_first_order_prompt_gen = FirstOrderPromptGeneration.from_llm(
        llm=llm,
        task_prompt_factory=str_task_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        llm_kwargs={"num_predict": num_predict},
        # verbose=1,
        # callbacks=[handler]
    )

    # Distribution Mutators
    mutator_estimation_of_distribution = EstimationOfDistributionMutation.from_llm(
        llm=llm,
        embeddings=embed_model,
        distance_metric=EmbeddingDistance.COSINE,
        task_prompt_factory=str_task_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        threshold=ed_threshold,
        llm_kwargs={"num_predict": num_predict},
    )
    mutator_eda_rank = EdaRankAndIndexMutation.from_llm(
        llm=llm,
        embeddings=embed_model,
        distance_metric=EmbeddingDistance.COSINE,
        task_prompt_factory=str_task_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        threshold=ed_threshold,
        fitness_scorer=fitness_scorer,
        llm_kwargs={"num_predict": num_predict},
    )
    mutator_lineage = LineageBasedMutation.from_llm(
        llm=llm,
        task_prompt_factory=str_task_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        llm_kwargs={"num_predict": num_predict},
    )

    # Hypermutations
    mutator_zero_order_hyper = ZeroOrderHypermutation.from_llm(
        task_prompt_factory=str_task_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        thinking_style_provider=thinking_style_provider,
        llm=llm,
        llm_kwargs={"num_predict": num_predict},
        # verbose=1,
        # callbacks=[handler]
    )
    mutator_first_order_hyper = FirstOrderHypermutation.from_llm(
        task_prompt_factory=str_task_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        thinking_style_provider=thinking_style_provider,
        llm=llm,
        llm_kwargs={"num_predict": num_predict},
        # verbose=1,
        # callbacks=[handler]
    )

    # Modifiers
    mutator_prompt_corssover = PromptCrossover(
        task_prompt_factory=str_task_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        fitness_scorer=fitness_scorer,
        probability_of_replacement=crossover_prob,
    )
    mutator_elite = AddElite(
        task_prompt_factory=str_task_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        fitness_scorer=fitness_scorer,
    )
    # Since no COT yet. No context shuffling

    # Initialize
    initializer = ZeroOrderInitialization.from_llm(
        problem_description_factory=str_problem_desc_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        task_prompt_factory=str_task_prompt_factory,
        thinking_style_provider=thinking_style_provider,
        mutation_prompt_provider=mutation_prompt_provider,
        llm=llm,
        n_members_per_unit=n_members_per_unit,
        llm_kwargs={"num_predict": num_predict},
    )
    pop_initializer = PositivePopulationInitialization(
        initializer=initializer,
        n_units=n_units,
        fitness_scorer=multiple_scorer,
        value=0.0,
    )

    reinit = ReplaceWithInit(
        fitness_scorer=multiple_scorer,
        value=0.0,
        initializer=initializer,
        mutation_prompt_factory=str_mutation_prompt_factory,
        task_prompt_factory=str_task_prompt_factory,
    )

    evolution_step = BinaryEvolution(
        fitness_scorer=multiple_scorer,
        pre_step_modifiers=[],
        mutators=[
            mutator_zero_order_prompt_gen,
            mutator_first_order_prompt_gen,
            mutator_estimation_of_distribution,
            mutator_eda_rank,
            mutator_lineage,
            mutator_zero_order_hyper,
            mutator_first_order_hyper,
        ],
        post_step_modifiers=[mutator_prompt_corssover, mutator_elite, reinit],
    )
    evolution = EvolutionExecutor(
        step=evolution_step, post_step_callback=[cb0, cb1, cb2, cb3]
    )

    return pop_initializer, evolution
