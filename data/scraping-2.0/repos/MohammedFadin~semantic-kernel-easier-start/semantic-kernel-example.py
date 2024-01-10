import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion # You can import OpenAI OpenAIChatCompletion
from plugins.OrchestratorPlugin.Orchestrator import Orchestrator # You can import your own plugin as well
#### ADD-HERE-STEP #### 
from plugins.YourPluginFolderName.YourPluginFileName import your_plugin_class_name # Importing a plugin from the plugins folder

async def main():
    
    #### MANDATORY-STEP #### Create a semantic kernel builder
    kernel = sk.Kernel()
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    kernel.add_chat_service("Your-Application-Name", AzureChatCompletion(deployment, endpoint, api_key)) #

    #### MANDATORY-STEP ####
    # Import a semantic plugin (previously called skills) from the plugins folder
    # This step is mandatory and allows you to add custom functionality to your chatbot
    plugins_directory = "./plugins" 
    kernel.import_semantic_skill_from_directory(plugins_directory, "OrchestratorPlugin")
    orchestrator_plugin = kernel.import_skill(Orchestrator(kernel), "OrchestratorPlugin")

    #### ADD-HERE-STEP #### 
    # Import your own custom plugin from the plugins folder
    # This step is optional and allows you to add additional functionality to your chatbot
    # Replace "your_native_plugin_folder_name" with the name of your plugin folder name
    # Replace "your_plugin_class_name" with the name of your plugin
    # Example: my_plugin = kernel.import_skill(my_plugin_class_name(), "MyPluginFolderName")
    # Note: Make sure your plugin is defined in a separate Python file in the plugins folder
    # and you have imported the plugin at the top of this file
    # Example: your_plugin_class_name() should be defined in a file called my_plugin_function.py
    # located in the plugins folder
    your_native_plugin_name = kernel.import_skill(your_plugin_class_name(), "your_native_plugin_folder_name")

    #### ADD-HERE-STEP ####
    # Pass user input to the orchestrator plugin and get the output
    # This step is mandatory and allows the chatbot to process user input and generate a response
    # Replace "kernel_output" with a variable name that describes the output of the orchestrator plugin
    # Replace "user_input" with the name of the variable that stores user input
    # Example: output = await kernel.run_async(orchestrator_plugin["RouteRequest"], input_str = user_input)
    # Note: The "RouteRequest" function is defined in the Orchestrator plugin and is responsible for
    # processing user input and generating a response (think of it as a middleware function)
    user_input = input("How can I help you?  ")
    kernel_output = await kernel.run_async(orchestrator_plugin["RouteRequest"], input_str = user_input)   
    print(your_input_to_kernel)

asyncio.run(main())