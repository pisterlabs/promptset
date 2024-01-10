from termcolor import colored
import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion # You can import OpenAI OpenAIChatCompletion
from plugins.StocksReaderPlugin.stocks_reader_plugin import Stocks # Importing a plugin from the plugins folder
from plugins.OrchestratorPlugin.Orchestrator import Orchestrator # Importing the Orchestrator plugin from the plugins folder

async def main():
    
    # Create a semantic kernel builder
    kernel = sk.Kernel()
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    kernel.add_chat_service("my_console_chat", AzureChatCompletion(deployment, endpoint, api_key))

    # Import a semantic plugin (previously called skills) from the plugins folder
    plugins_directory = "./plugins"
    kernel.import_semantic_skill_from_directory(plugins_directory, "OrchestratorPlugin")
    orchestrator_plugin = kernel.import_skill(Orchestrator(kernel), "OrchestratorPlugin")

    # Import a native plugin like a stocks plugin to fetch stock data from Dubai Financial Market
    kernel.import_skill(Stocks(), "StocksReaderPlugin")

    # Run the kernel to load the plugins and output the response in a chat console style
    while True:
        try:
            # Run the prompt and get the user input
            console_user_input = input("How can I help you? \nUser:")
            kernel_output = await kernel.run_async(
                orchestrator_plugin["RouteRequest"], input_str = console_user_input
                )
            print(colored(kernel_output, "blue"))
            if console_user_input == "exit":
                break
        except KeyboardInterrupt:
            print("Bye!")
            break

asyncio.run(main())