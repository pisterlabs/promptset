"""
GPTenterprise is an AI driven enterprise.
"""
import os
import sys
import asyncio
import json
import openai
from dotenv import load_dotenv
from gpt_enterprise.enterprise import Enterprise


def main():
    load_dotenv(sys.argv[1])

    interactive = os.getenv("INTERACTIVE") == "yes"
    keyfile = os.getenv("KEYFILE")
    company_name = os.getenv("COMPANY_NAME")
    guidelines = os.getenv("CEO_GUIDELINES")
    output_directory = os.getenv("OUTPUT_DIRECTORY")
    manager_retry = int(os.getenv("MANAGER_RETRY"))
    local_ai_url = os.getenv("LOCAL_AI_URL")

    if local_ai_url:
        # Change api url to LoacalAi one
        openai.api_base = local_ai_url
        openai.api_key = "sx-xxx"
        OPENAI_API_KEY = "sx-xxx"
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    else:
        # Initialize openai api_key
        with open("./openai_key.txt", "r") as file:
            openai.api_key = (file.read()).strip()

    # Interactive mode, change variables with user input
    if interactive:
        keyfile = input(f"Path of OpenAI key file ? {keyfile}") or keyfile
        print(f"\n \U0001F511 \t {keyfile} \n")
        company_name = (
            input(f"What is your company name ? {company_name}") or company_name
        )
        print(f"\n \U0001F3E2 \t {company_name} \n")
        output_directory = (
            input(f"Where do you want to push the result ? {output_directory}")
            or output_directory
        )
        print(f"\n \U0001F5C2 \t {output_directory} \n")
        manager_retry = int(
            input(
                f"How many times the manager will retry to do the plan ? {manager_retry}"
            )
            or manager_retry
        )
        print(f"\n \U0001F9EC \t {manager_retry} \n")
        guidelines = (
            input(
                f"What are your guidelines ? If not provided, config guidelines will be used. \n"
            )
            or guidelines
        )
        print(f"\n \U0001F489 \t {guidelines} \n")

    # Create the enterprise
    enterprise = Enterprise(
        company_name=company_name,
        guidelines=guidelines,
        output_directory=output_directory,
        manager_retry=manager_retry,
        interactive=interactive,
    )

    # Run the enterprise and go into production !
    production = asyncio.run(enterprise.run_enterprise())
    # Write the production in a JSON file
    with open(
        os.path.join(output_directory, f"production_{company_name}.json"), "w"
    ) as file:
        file.write(json.dumps(production, indent=4))


if __name__ == "__main__":
    main()
