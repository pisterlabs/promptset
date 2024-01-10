import argparse
import json
from discordai import __version__ as version
from discordai import config as configuration
from discordai import template
from discordai import bot
from discordai_modelizer import customize
from discordai_modelizer import openai as openai_wrapper


def discordai():
    config = configuration.get()
    parser = argparse.ArgumentParser(
        prog="discordai", description="discordAI CLI"
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"discordai {version}"
    )
    command = parser.add_subparsers(dest="command")

    bot_cmds = command.add_parser("bot", description="Commands related to your discord bot")
    model = command.add_parser("model", description="Commands related to your openAI models")
    job = command.add_parser("job", description="Commands related to your openAI jobs")
    config_cmds = command.add_parser("config", description="View and modify your config")

    bot_cmds_subcommand = bot_cmds.add_subparsers(dest="subcommand")
    model_subcommand = model.add_subparsers(dest="subcommand")
    job_subcommand = job.add_subparsers(dest="subcommand")
    config_cmds_subcommand = config_cmds.add_subparsers(dest="subcommand")

    bot_cmds_start = bot_cmds_subcommand.add_parser(
        "start", description="Start your discord bot"
    )
    bot_cmds_start.add_argument(
        "--sync",
        action='store_true',
        required=False,
        dest='sync',
        help="Sync discord commands gloablly on start up",
    )

    bot_cmds_commands = bot_cmds_subcommand.add_parser(
        "commands", description="Manage your discord bot's slash commands"
    )
    bot_cmds_commands_subcommand = bot_cmds_commands.add_subparsers(dest="subsubcommand")

    new_cmd = bot_cmds_commands_subcommand.add_parser(
        "new", description="Create a new slash command for your bot that will use a customized model for completions"
    )
    new_cmd_required_named = new_cmd.add_argument_group(
        "required named arguments"
    )
    new_cmd_required_named.add_argument(
        "-n", "--command-name",
        type=str,
        dest='command_name',
        help="The name you want to use for the command",
    )
    new_cmd_required_named.add_argument(
        "-i", "--model-id",
        type=str,
        dest='model_id',
        help="The ID of the customized model for the slash command to use",
    )
    new_cmd_optional_named = new_cmd.add_argument_group("optional named arguments")
    new_cmd_optional_named.add_argument(
        "-o", "--openai-key",
        type=str,
        default=config["openai_key"],
        required=False,
        dest='openai_key',
        help="The openAI key associated with the model being used: DEFAULT=config.openai_key",
    )
    new_cmd_optional_named.add_argument(
        "-t", "--temp-default",
        type=float,
        default=1,
        required=False,
        dest='temp_default',
        help="The default temperature to use for completions: DEFAULT=1",
    )
    new_cmd_optional_named.add_argument(
        "-p", "--pres-default",
        type=float,
        default=0,
        required=False,
        dest='pres_default',
        help="The default presence penalty to use for completions: DEFAULT=0",
    )
    new_cmd_optional_named.add_argument(
        "-f", "--freq-default",
        type=float,
        default=0,
        required=False,
        dest='freq_default',
        help="The default frequency penalty to use for completions: DEFAULT=0",
    )
    new_cmd_optional_named.add_argument(
        "-m", "--max-tokens-default",
        type=int,
        default=125,
        required=False,
        dest='max_tokens_default',
        help="The default max tokens to use for completions: DEFAULT=125",
    )
    new_cmd_optional_named.add_argument(
        "--stop-default",
        action='store_true',
        required=False,
        dest='stop_default',
        help="Set the stop option to use for completions to True",
    )
    new_cmd_optional_named.add_argument(
        "--bold_default",
        action='store_true',
        required=False,
        dest='bold_default',
        help="Set the bolden option for prompts to True",
    )

    delete_cmd = bot_cmds_commands_subcommand.add_parser(
        "delete", description="Delete a slash command from your bot"
    )
    delete_cmd_required_named = delete_cmd.add_argument_group(
        "required named arguments"
    )
    delete_cmd_required_named.add_argument(
        "-n", "--command-name",
        type=str,
        dest='command_name',
        help="The name of the slash command you want to delete",
    )

    model_list = model_subcommand.add_parser(
        "list", description="List your openAi customized models"
    )
    model_list.add_argument(
        "-o", "--openai-key",
        type=str,
        default=config["openai_key"],
        required=False,
        dest='openai_key',
        help="The openAI API key to list the models for: DEFAULT=config.openai_key",
    )
    model_list.add_argument(
        "--simple",
        action='store_true',
        required=False,
        dest='simple',
        help="Simplify the output to just the model name, job id, and status",
    )

    model_create = model_subcommand.add_parser(
        "create",
        description="Create a new openAI customized model by downloading the specified chat logs, parsing them into a usable dataset, and then training a customized model using openai")
    model_create_required_named = model_create.add_argument_group(
        "required named arguments"
    )
    model_create_required_named.add_argument(
        "-c", "--channel",
        type=str,
        dest='channel',
        help="The ID of the discord channel you want to use",
    )
    model_create_required_named.add_argument(
        "-u", "--user",
        type=str,
        dest='user',
        help="The username of the discord user you want to use",
    )
    model_create_optional_named = model_create.add_argument_group("optional named arguments")
    model_create_optional_named.add_argument(
        "-o", "--openai-key",
        type=str,
        default=config["openai_key"],
        required=False,
        dest='openai_key',
        help="The openAI API key to use to create the model: DEFAULT=config.openai_key",
    )
    model_create_optional_named.add_argument(
        "-b", "--base-model",
        choices=["davinci", "curie", "babbage", "ada", "none"],
        default="none",
        required=False,
        dest='base_model',
        help="The base model to use for customization. If none, then skips training step: DEFAULT=none",
    )
    model_create_optional_named.add_argument(
        "--ttime", "--thought-time",
        type=int,
        default=10,
        required=False,
        dest='thought_time',
        help="The max amount of time in seconds to consider two individual messages to be part of the same \"thought\": DEFAULT=10",
    )
    model_create_optional_named.add_argument(
        "--tmax", "--thought-max",
        type=int,
        default=None,
        required=False,
        dest='thought_max',
        help="The max in words length of each thought: DEFAULT=None",
    )
    model_create_optional_named.add_argument(
        "--tmin", "--thought-min",
        type=int,
        default=4,
        required=False,
        dest='thought_min',
        help="The minimum in words length of each thought: DEFAULT=4",
    )
    model_create_optional_named.add_argument(
        "-m", "--max-entries",
        type=int,
        default=1000,
        required=False,
        dest='max_entries',
        help="The max amount of entries that may exist in the dataset: DEFAULT=1000",
    )
    model_create_optional_named.add_argument(
        "-r", "--reduce-mode",
        choices=["first", "last", "middle", "even"],
        default="even",
        required=False,
        dest='reduce_mode',
        help="The method to reduce the entry count of the dataset: DEFAULT=even",
    )
    model_create_optional_named.add_argument(
        "--dirty",
        action='store_false',
        required=False,
        dest='dirty',
        help="Skip the clean up step for outputted files",
    )
    model_create_optional_named.add_argument(
        "--redownload",
        action='store_true',
        required=False,
        dest='redownload',
        help="Redownload the discord chat logs",
    )

    model_delete = model_subcommand.add_parser(
        "delete", description="Delete an openAI customized model"
    )
    model_delete_required_named = model_delete.add_argument_group(
        "required named arguments"
    )
    model_delete_required_named.add_argument(
        "-m", "--model-id",
        type=str,
        dest='model_id',
        help="Target model id",
    )
    model_delete_optional_named = model_delete.add_argument_group("optional named arguments")
    model_delete_optional_named.add_argument(
        "-o", "--openai-key",
        type=str,
        default=config["openai_key"],
        required=False,
        dest='openai_key',
        help="The openAI API key associated with the model to delete: DEFAULT=config.openai_key",
    )

    job_list = job_subcommand.add_parser(
        "list", description="List your openAI customization jobs"
    )
    job_list.add_argument(
        "-o", "--openai-key",
        type=str,
        default=config["openai_key"],
        required=False,
        dest='openai_key',
        help="The openAI API key to list the jobs for: DEFAULT=config.openai_key",
    )
    job_list.add_argument(
        "--simple",
        action='store_true',
        required=False,
        dest='simple',
        help="Simplify the output to just the model name, job id, and status",
    )

    job_follow = job_subcommand.add_parser(
        "follow", description="Follow an openAI customization job"
    )
    job_follow_required_named = job_follow.add_argument_group(
        "required named arguments"
    )
    job_follow_required_named.add_argument(
        "-j", "--job-id",
        type=str,
        dest='job_id',
        help="Target job id",
    )
    job_follow_optional_named = job_follow.add_argument_group("optional named arguments")
    job_follow_optional_named.add_argument(
        "-o", "--openai-key",
        type=str,
        default=config["openai_key"],
        required=False,
        dest='openai_key',
        help="The openAI API key associated with the job to follow: DEFAULT=config.openai_key",
    )

    job_status = job_subcommand.add_parser(
        "status", description="Get an openAI customization job's status"
    )
    job_status_required_named = job_status.add_argument_group(
        "required named arguments"
    )
    job_status_required_named.add_argument(
        "-j", "--job-id",
        type=str,
        dest='job_id',
        help="Target job id",
    )
    job_status_optional_named = job_status.add_argument_group("optional named arguments")
    job_status_optional_named.add_argument(
        "-o", "--openai-key",
        type=str,
        default=config["openai_key"],
        required=False,
        dest='openai_key',
        help="The openAI API key associated with the job to see the status for: DEFAULT=config.openai_key",
    )
    job_status_optional_named.add_argument(
        "--events",
        action='store_true',
        required=False,
        dest='events',
        help="Simplify the output to just the event list",
    )

    job_cancel = job_subcommand.add_parser(
        "cancel", description="Cancel an openAI customization job"
    )
    job_cancel_required_named = job_cancel.add_argument_group(
        "required named arguments"
    )
    job_cancel_required_named.add_argument(
        "-j", "--job-id",
        type=str,
        dest='job_id',
        help="Target job id",
    )
    job_cancel_optional_named = job_cancel.add_argument_group("optional named arguments")
    job_cancel_optional_named.add_argument(
        "-o", "--openai-key",
        type=str,
        default=config["openai_key"],
        required=False,
        dest='openai_key',
        help="The openAI API key associated with the job to cancel: DEFAULT=config.openai_key",
    )

    config_bot_token = config_cmds_subcommand.add_parser(
        "bot-token", description="Get or set your discord bot token"
    )
    config_bot_token.add_argument(
        "-t", "--set-token",
        type=str,
        dest='new_token',
        help="Discord bot token",
    )

    config_openai_key = config_cmds_subcommand.add_parser(
        "openai-key", description="Get or set your openaAI API key"
    )
    config_openai_key.add_argument(
        "-k", "--set-key",
        type=str,
        dest='new_key',
        help="OpenAI API key",
    )

    args = parser.parse_args()
    if args.command == "bot":
        if args.subcommand == "start":
            bot.start_bot(config, args.sync)
        if args.subcommand == "commands":
            if args.subsubcommand == "new":
                template.gen_new_command(args.model_id, args.command_name, args.temp_default, args.pres_default,
                                         args.freq_default, args.max_tokens_default, args.stop_default, args.openai_key,
                                         args.bold_default)
            elif args.subsubcommand == "delete":
                template.delete_command(args.command_name)
    elif args.command == "model":
        if args.subcommand == "list":
            openai_wrapper.list_models(args.openai_key, args.simple)
        if args.subcommand == "create":
            customize.create_model(config["token"], args.openai_key, args.channel, args.user,
                                   thought_time=args.thought_time, thought_max=args.thought_max, thought_min=args.thought_min,
                                   max_entry_count=args.max_entries, reduce_mode=args.reduce_mode, base_model=args.base_model, 
                                   clean=args.dirty, redownload=args.redownload)
        if args.subcommand == "delete":
            openai_wrapper.delete_model(args.openai_key, args.model_id)
    elif args.command == "job":
        if args.subcommand == "list":
            openai_wrapper.list_jobs(args.openai_key, args.simple)
        if args.subcommand == "follow":
            openai_wrapper.follow_job(args.openai_key, args.job_id)
        if args.subcommand == "status":
            openai_wrapper.get_status(args.openai_key, args.job_id, args.events)
        if args.subcommand == "cancel":
            openai_wrapper.cancel_job(args.openai_key, args.job_id)
    elif args.command == "config":
        if args.subcommand == "bot-token":
            if args.new_token:
                print(f"Old discord bot token: {config['token']}")
                configuration.save(json.dumps(dict(token=args.new_token, openai_key=config["openai_key"])))
                config = configuration.get()
            print(f"Current discord bot token: {config['token']}")
        elif args.subcommand == "openai-key":
            if args.new_key:
                print(f"Old openAi API key: {config['openai_key']}")
                configuration.save(json.dumps(dict(token=config["token"], openai_key=args.new_key)))
                config = configuration.get()
            print(f"Current openAi API key: {config['openai_key']}")
        else:
            print(f"Current discord bot token: {config['token']}")
            print(f"Current openAi API key: {config['openai_key']}")


if __name__ == "__main__":
    try:
        discordai()
    except KeyboardInterrupt:
        print("Program interrupted by user. Exiting...")
