import asyncio
from tkinter import Variable

from numpy import var
from Plugins.orchestratorPlugin.OrchestratorPlugin import Orchestrator
import semantic_kernel as sk
from Plugins.MathPlugin.Math import Math
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion

async def main():
    kernel = sk.Kernel()
    useAzureOpenAI = False

    if useAzureOpenAI:
        deployment, api, endpoint = sk.azure_openai_settings_from_dot_env()
        kernel.add_text_completion_service("azureioenai", AzureChatCompletion)
    else:
        api_key, org_id = sk.openai_settings_from_dot_env()
        kernel.add_text_completion_service("openai", OpenAIChatCompletion("gpt-3.5-turbo-0301", api_key, org_id))
    print("you made a kernel!")

    math_plugin  = kernel.import_skill(Math(), skill_name="MathPlugin")
    orchestrator_plugin = kernel.import_skill(
        Orchestrator(kernel = kernel),
        skill_name="orchestratorPlugin"
    )


    # variables for multiply function: give two inputs
    variables = sk.ContextVariables()
    # using contextvariable object and variables property pass into the kernel
    variables["input"] = "12.34"
    variables["number2"] = "56.78"

    result = await kernel.run_async(
        math_plugin["Sqrt"],
        input_str="25")
    
    result_multiple = await kernel.run_async(
        math_plugin["Multiply"],
        input_vars=variables)

    print(result)
    print(result_multiple)

    # the following code is for using orchestrator plugin 
        # Make a request that runs the Sqrt function.
    result_with_one = await kernel.run_async(
        orchestrator_plugin["RouteRequest"],
        input_str="What is the square root of 634?",
    )
    print(result1)

    # Make a request that runs the Multiply function.
    result_with_two = await kernel.run_async(
        orchestrator_plugin["RouteRequest"],
        input_str="What is 12.34 times 56.78?",
    )

asyncio.run(main())