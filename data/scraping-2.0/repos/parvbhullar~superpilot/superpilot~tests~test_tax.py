import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from superpilot.framework.abilities import (
    TextSummarizeAbility,
)
from superpilot.core.resource.model_providers import (
    ModelProviderName,
    OpenAIProvider,
    AnthropicApiProvider,
    AnthropicModelName,
    OllamaModelName,
    OllamaApiProvider,
    OpenAIModelName,
)
from superpilot.core.context.schema import Context
from superpilot.core.ability.super import SuperAbilityRegistry
from superpilot.core.pilot.task.simple import SimpleTaskPilot
from superpilot.examples.ed_tech.ag_question_solver_ability import AGQuestionSolverAbility
from superpilot.examples.tax.gstr1_data_transformer import GSTR1DataTransformerPrompt
from superpilot.examples.pilots.tasks.super import SuperTaskPilot
from superpilot.core.planning.schema import Task

ALLOWED_ABILITY = {
    # SearchAndSummarizeAbility.name(): SearchAndSummarizeAbility.default_configuration,
    # TextSummarizeAbility.name(): TextSummarizeAbility.default_configuration,
    AGQuestionSolverAbility.name(): AGQuestionSolverAbility.default_configuration,
}
from superpilot.tests.test_env_simple import get_env
from superpilot.core.pilot import SuperPilot
from superpilot.core.configuration import get_config
from superpilot.core.planning.settings import (
    LanguageModelConfiguration,
    LanguageModelClassification,
)

# Flow executor -> Context
#
# Context -> Search & Summarise [WebSearch, KnowledgeBase, News] - parrallel] -> Context
# Context -> Analysis[FinancialAnalysis] -> Context
# Context -> Write[TextStream, PDF, Word] -> Context
# Context -> Respond[Twitter, Email, Stream]] -> Context
# Context -> Finalise[PDF, Word] -> Context

def call_open_ai():
    # from openai import OpenAI
    import openai
    # client = OpenAI()

    response = openai.ChatCompletion.create(
        model=OpenAIModelName.GPT4_TURBO,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]
    )
    print(response)


async def test_pilot():
    # call_open_ai()
    # exit(0)
    # query = "How to file the GSTR1"
    query = "What is the weather in Mumbai"
    query = """
    Supplier GSTIN	Invoice Status	Transaction Number	Supply Type	Invoice No.	Invoice Date	Invoice Type	Note Number	Note Date	HSN/SAC	Item Description	Quantity	UQC (unit of measure)	Invoice Value	Note Value	Taxable Value	GST Rate	IGST Amount	CGST Amount	SGST/UTGST Amount	CESS Amount	Revenue Account	Customer GSTIN	Customer Name	Place of Supply	Export Type	Shipping Bill Number	Shipping Bill Date	Port Number	GSTR1 Return Period	3B Auto-fill Period	Location	isamended	Document Number	Document Date	Reverse Charge	Original Invoice Number	Original Invoice Date	Original Month	Amortised cost
27GSPMH0591G1ZK	Add	23210417	Normal	800/50	29-08-2021	Tax Invoice			0208		25.56	KGS	3068		45600.56	18	8208.1008	0	0	25.65	21.8500.111.150430.0000.000000.0000.000000.000000.00000000	33GSPTN9511G3Z3	IPM INDIA WHOLESALE TRADING PVT LTD	9					08-2021	082021	Delhi		GHTY/1200-005	15-02-2022					
27GSPMH0591G1ZK	Add	23210417	Normal	800/51	29-08-2021	Tax Invoice			0208		25.56	KGS	3068		85900.56	18	0	7731.0504	7731.0504	50.56	21.8500.111.150430.0000.000000.0000.000000.000000.00000000	33GSPTN9511G3Z3	IPM INDIA WHOLESALE TRADING PVT LTD	27					08-2021	082021	Delhi		GHTY/1200-005	15-02-2022					
27GSPMH0591G1ZK	Add	23210418	Normal	800/52	29-08-2021	Tax Invoice			9102		35.65	NOS	28910		256890.67	12	30826.8804	0	0	50.56	21.8500.111.150430.0000.000000.0000.000000.000000.00000000		IPM INDIA WHOLESALE TRADING PVT LTD	9					08-2021	082021	Delhi		GHTY/1200-005	15-02-2022					
27GSPMH0591G1ZK	Add	23210419	Normal	800/53	29-08-2021	Tax Invoice			0208		27.56	KGS	10620		236800.65	5	11840.0325	0	0	0	21.8500.111.150430.0000.000000.0000.000000.000000.00000000		IPM INDIA WHOLESALE TRADING PVT LTD	9					08-2021	082021	Delhi		GHTY/1200-005	15-02-2022					
27GSPMH0591G1ZK	Add	23210419	Normal	800/54	29-08-2021	Tax Invoice			0208		27.56	KGS	10620		237800.65	5	0	5945.01625	5945.01625	78.89	21.8500.111.150430.0000.000000.0000.000000.000000.00000000		IPM INDIA WHOLESALE TRADING PVT LTD	27					08-2021	082021	Delhi		GHTY/1200-005	15-02-2022					
    """

    context = Context()

    config = get_config()
    env = get_env({})

    print(config.openai_api_key)
    # Load Model Providers
    open_ai_provider = OpenAIProvider.factory(config.openai_api_key)
    anthropic_provider = AnthropicApiProvider.factory(config.anthropic_api_key)
    ollama_provider = OllamaApiProvider.factory(config.anthropic_api_key)
    model_providers = {ModelProviderName.OPENAI: open_ai_provider, ModelProviderName.ANTHROPIC: anthropic_provider}
    # model_providers = {ModelProviderName.OLLAMA: ollama_provider}

    # Load Prompt Strategy
    # SimpleTaskPilot.default_configuration.models = {
    #     LanguageModelClassification.FAST_MODEL: LanguageModelConfiguration(
    #         model_name=AnthropicModelName.CLAUD_2_INSTANT,
    #         provider_name=ModelProviderName.ANTHROPIC,
    #         temperature=1,
    #     ),
    #     LanguageModelClassification.SMART_MODEL: LanguageModelConfiguration(
    #         model_name=AnthropicModelName.CLAUD_2,
    #         provider_name=ModelProviderName.ANTHROPIC,
    #         temperature=0.9,
    #     ),
    # }
    SimpleTaskPilot.default_configuration.models = {
        LanguageModelClassification.FAST_MODEL: LanguageModelConfiguration(
            model_name=OpenAIModelName.GPT3,
            provider_name=ModelProviderName.OPENAI,
            temperature=0.2,
        ),
        LanguageModelClassification.SMART_MODEL: LanguageModelConfiguration(
            model_name=OpenAIModelName.GPT4,
            provider_name=ModelProviderName.OPENAI,
            temperature=0.2,
        ),
    }

    task_pilot = SimpleTaskPilot.create(
        GSTR1DataTransformerPrompt.default_configuration,
        model_providers=model_providers,
        smart_model_name=OpenAIModelName.GPT4_TURBO,
        fast_model_name=OpenAIModelName.GPT4_TURBO,
    )

    print("***************** Executing SimplePilot ******************************\n")
    response = await task_pilot.execute(query, context)
    print(response)
    print("***************** Executing SimplePilot Completed ******************************\n")

    exit(0)
    # # Load Prompt Strategy
    # super_prompt = SimplePrompt.factory()
    # # Load Abilities
    # prompt = super_prompt.build_prompt(query)
    # print(prompt)

    super_ability_registry = SuperAbilityRegistry.factory(env, ALLOWED_ABILITY)
    pilot = SuperTaskPilot(super_ability_registry, model_providers)
    task = Task.factory(query)
    context = await pilot.execute(task, context)
    print(context)
    exit(0)
    #
    super_ability_registry = SuperAbilityRegistry.factory(env, ALLOWED_ABILITY)
    #
    # search_step = SuperTaskPilot(super_ability_registry, model_providers)

    planner = env.get("planning")
    ability_registry = env.get("ability_registry")

    # user_configuration = {}
    # pilot_settings = SuperPilot.compile_settings(client_logger, user_configuration)

    # Step 2. Provision the environment.
    # environment_workspace = SuperPilot.provision_environment(environment_settings, client_logger)



    exit(0)
    user_objectives = "What is the weather in Mumbai"
    user_objectives = query
    # SuperPilot.default_settings.configuration
    pilot_settings = SuperPilot.default_settings
    pilot = SuperPilot(pilot_settings, super_ability_registry, planner, env)
    print(await pilot.initialize(user_objectives))
    print(
        "***************** Pilot Initiated - Planing Started ******************************\n"
    )
    print(await pilot.plan())
    print(
        "***************** Pilot Initiated - Planing Completed ******************************\n"
    )
    while True:
        print(
            "***************** Pilot Started - Exec Started ******************************\n"
        )
        # current_task, next_ability = await pilot.determine_next_ability(plan)
        # print(parse_next_ability(current_task, next_ability))
        # user_input = click.prompt(
        #     "Should the pilot proceed with this ability?",
        #     default="y",
        # )
        # if not next_ability["next_ability"]:
        #     print("Agent is done!", "No Next Ability Found")
        #     break
        # ability_result = await pilot.execute_next_ability(user_input)
        # print(parse_ability_result(ability_result))
        break


if __name__ == "__main__":
    asyncio.run(test_pilot())
