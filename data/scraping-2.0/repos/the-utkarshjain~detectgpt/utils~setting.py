import os
import json
import datetime

def initial_setup(args, config):
    API_TOKEN_COUNTER = 0

    if args.openai_model is not None:
        import openai
        assert args.openai_key is not None, "Must provide OpenAI API key as --openai_key"
        openai.api_key = args.openai_key

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    # define SAVE_FOLDER as the timestamp - base model name - mask filling model name
    # create it if it doesn't exist
    precision_string = "int8" if args.int8 else ("fp16" if args.half else "fp32")
    sampling_string = "top_k" if args.do_top_k else ("top_p" if args.do_top_p else "temp")
    output_subfolder = f"{args.output_name}/" if args.output_name else ""
    if args.openai_model is None:
        base_model_name = args.base_model_name.replace('/', '_')
    else:
        base_model_name = "openai-" + args.openai_model.replace('/', '_')
    scoring_model_string = (f"-{args.scoring_model_name}" if args.scoring_model_name else "").replace('/', '_')
    SAVE_FOLDER = f"tmp_results/{output_subfolder}{base_model_name}{scoring_model_string}-{args.mask_filling_model_name}-{sampling_string}/{START_DATE}-{START_TIME}-{precision_string}-{args.pct_words_masked}-{args.n_perturbation_rounds}-{args.dataset}-{args.n_samples}"
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_FOLDER)}")

    # write args to file
    with open(os.path.join(SAVE_FOLDER, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    
    config["START_DATE"] = START_DATE
    config["START_TIME"] = START_TIME
    config["base_model_name"] = base_model_name
    config["SAVE_FOLDER"] = SAVE_FOLDER
    config["API_TOKEN_COUNTER"] = API_TOKEN_COUNTER

def set_experiment_config(args, config):
    """
    Parses the runtime arguments for setting the experiment configuration.
    """
    cache_dir = args.cache_dir
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    config["mask_filling_model_name"] = args.mask_filling_model_name
    config["n_samples"] = args.n_samples
    config["batch_size"] = args.batch_size
    config["n_perturbation_list"] = [int(x) for x in args.n_perturbation_list.split(",")]
    config["n_perturbation_rounds"] = args.n_perturbation_rounds
    config["n_similarity_samples"] = args.n_similarity_samples
    config["cache_dir"] = args.cache_dir