import openai
from utils import (
    read_dataset,
    read_raw_tufano_dataset_from_csv,
    get_env_variable,
    get_EM,
    get_few_shot_predictions_from_openai_and_write_to_file,
    get_bleu_and_codebleu,
apply_heuristic_in_file
)

if __name__ == "__main__":
    openai.api_key = get_env_variable("OPENAI_API_KEY")
    OUTPUT_DIRECTORY = "few_shot_outputs"
    TUFANO_RAW_TRAIN_DATASET_FILE_PATH = "datasets/tufano/raw_train.csv"
    TUFANO_RAW_TEST_DATASET_FILE_PATH = "datasets/tufano/raw_test.csv"

    R4R_TRAIN_DATASET_SRC_FILE_PATH = "datasets/R4R/train_CC_src.txt"
    R4R_TRAIN_DATASET_TGT_FILE_PATH = "datasets/R4R/train_CC_tgt.txt"
    R4R_TEST_DATASET_SRC_FILE_PATH = "datasets/R4R/test_CC_src.txt"
    R4R_TEST_DATASET_TGT_FILE_PATH = "datasets/R4R/test_CC_tgt.txt"

    START_INDEX = 1228
    END_INDEX = 1500

    # test_code_reviews, test_buggy_codes, test_target_codes = read_raw_tufano_dataset_from_csv(
    #     TUFANO_RAW_TEST_DATASET_FILE_PATH
    # )
    # train_code_reviews, train_buggy_codes, train_target_codes = read_raw_tufano_dataset_from_csv(
    #     TUFANO_RAW_TRAIN_DATASET_FILE_PATH
    # )

    # get_few_shot_predictions_from_openai_and_write_to_file(
    #     prediction_file_path=f"{OUTPUT_DIRECTORY}/few_shot_tufano_predictions_raw_no_heuristic.txt",
    #     ground_truth_path=f"{OUTPUT_DIRECTORY}/few_shot_tufano_ground_truths_raw_no_heuristic.txt",
    #     train_dataset=(train_code_reviews, train_buggy_codes, train_target_codes),
    #     test_dataset=(test_code_reviews, test_buggy_codes, test_target_codes),
    #     top_k=3
    # )

    # train_code_reviews, train_buggy_codes, train_target_codes = read_dataset(
    #     dataset_name="R4R",
    #     source_file_path=R4R_TRAIN_DATASET_SRC_FILE_PATH,
    #     target_file_path=R4R_TRAIN_DATASET_TGT_FILE_PATH
    # )
    #
    # test_code_reviews, test_buggy_codes, test_target_codes = read_dataset(
    #     dataset_name="R4R",
    #     source_file_path=R4R_TEST_DATASET_SRC_FILE_PATH,
    #     target_file_path=R4R_TEST_DATASET_TGT_FILE_PATH
    # )
    #
    # get_few_shot_predictions_from_openai_and_write_to_file(
    #     prediction_file_path=f"{OUTPUT_DIRECTORY}/few_shot_r4r_predictions_raw_no_heuristic.txt",
    #     ground_truth_path=f"{OUTPUT_DIRECTORY}/few_shot_r4r_ground_truths_raw_no_heuristic.txt",
    #     train_dataset=(train_code_reviews, train_buggy_codes, train_target_codes),
    #     test_dataset=(test_code_reviews, test_buggy_codes, test_target_codes),
    #     top_k=3,
    #     start_index=START_INDEX,
    # )

    get_EM(
        f"{OUTPUT_DIRECTORY}/few_shot_r4r_predictions_raw.txt",
        f"{OUTPUT_DIRECTORY}/few_shot_r4r_ground_truths_raw.txt",
        "R4R"
    )

    get_bleu_and_codebleu(
        f"{OUTPUT_DIRECTORY}/few_shot_r4r_predictions_raw.txt",
        f"{OUTPUT_DIRECTORY}/few_shot_r4r_ground_truths_raw.txt"
    )

    apply_heuristic_in_file(f"{OUTPUT_DIRECTORY}/few_shot_r4r_predictions_raw.txt")

    get_EM(
        f"{OUTPUT_DIRECTORY}/few_shot_r4r_predictions_raw_applied_heuristics.txt",
        f"{OUTPUT_DIRECTORY}/few_shot_r4r_ground_truths_raw_target_only.txt",
        "R4R"
    )

    get_bleu_and_codebleu(
        f"{OUTPUT_DIRECTORY}/few_shot_r4r_predictions_raw_applied_heuristics.txt",
        f"{OUTPUT_DIRECTORY}/few_shot_r4r_ground_truths_raw.txt",
    )

    get_EM(
        f"{OUTPUT_DIRECTORY}/few_shot_tufano_predictions_raw.txt",
        f"{OUTPUT_DIRECTORY}/few_shot_tufano_ground_truths_raw.txt",
        "tufano"
    )

    get_bleu_and_codebleu(
        f"{OUTPUT_DIRECTORY}/few_shot_tufano_predictions_raw.txt",
        f"{OUTPUT_DIRECTORY}/few_shot_tufano_ground_truths_raw.txt"
    )

    apply_heuristic_in_file(f"{OUTPUT_DIRECTORY}/few_shot_tufano_predictions_raw.txt")

    get_EM(
        f"{OUTPUT_DIRECTORY}/few_shot_tufano_predictions_raw_applied_heuristics.txt",
        f"{OUTPUT_DIRECTORY}/few_shot_tufano_ground_truths_raw.txt",
        "tufano"
    )

    get_bleu_and_codebleu(
        f"{OUTPUT_DIRECTORY}/few_shot_tufano_predictions_raw_applied_heuristics.txt",
        f"{OUTPUT_DIRECTORY}/few_shot_tufano_ground_truths_raw.txt",
    )

