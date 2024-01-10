import json
import os
from typing import Any, Dict
import gitreview_gpt.prompt as prompt
import gitreview_gpt.formatter as formatter
import gitreview_gpt.utils as utils
import gitreview_gpt.request as request


# Retrieve review from openai completions api
# Process response and send repair request if json has invalid format
def request_review(
    api_key, code_to_review, gpt_model, file_name=None
) -> Dict[str, Any] | None:
    max_tokens = gpt_model.value - utils.count_tokens(
        json.dumps(prompt.get_review_prompt(code_to_review, gpt_model.value, gpt_model))
    )
    payload = prompt.get_review_prompt(code_to_review, max_tokens, gpt_model)

    spinner_text = "🔍 Reviewing"
    if file_name is not None:
        spinner_text += f" {utils.get_bold_text(file_name)}"
    spinner_text += "..."

    review_result = request.send_request(api_key, payload, spinner_text)
    if not review_result:
        return None
    try:
        review_json = formatter.parse_review_result(review_result)
    except ValueError:
        try:
            # Try to parse review result from marldown code block
            review_json = formatter.parse_review_result(
                formatter.extract_content_from_markdown_code_block(review_result)
            )
        except ValueError:
            try:
                # Try to repair truncated review result
                review_json = formatter.parse_review_result(
                    utils.repair_truncated_json(review_result)
                )
            except ValueError as e:
                try:
                    print("Review result has invalid format. It will be repaired.")
                    payload = prompt.get_review_repair_prompt(
                        review_result, e, max_tokens, gpt_model
                    )
                    review_result = request.send_request(
                        api_key, payload, "🔧 Repairing..."
                    )
                    review_json = formatter.parse_review_result(
                        formatter.extract_content_from_markdown_code_block(
                            review_result
                        )
                    )
                except ValueError:
                    print("💥 Review result could not be repaired.")
                    print(review_result)
                    print(
                        "Feel free to create an issue at https://github.com/fynnfluegge/codereview-agi/issues"
                    )
                    return None

    return review_json


# Retrieve code changes from openai completions api
# for one specific file with the related review
def apply_review(
    api_key, absolute_file_path, review_json, selection_marker_chunks: Dict, gpt_model
):
    try:
        with open(absolute_file_path, "r") as file:
            file_name = os.path.basename(file.name)
            programming_language = utils.get_programming_language(file.name)
            file_content = file.read()
            payload = {
                "code": file_content,
                "reviews": formatter.get_review_suggestions_per_file_payload_from_json(
                    review_json
                ),
            }
            prompt_payload = prompt.get_apply_review_for_file_prompt(
                file_content,
                json.dumps(payload["reviews"]),
                gpt_model.value,
                programming_language,
                gpt_model,
            )
            tokens = utils.count_tokens(json.dumps(prompt_payload))
            # tokens for file content and review suggestions are greater than threshold
            # split requests into code chunks by selection markers
            if tokens > gpt_model.value / 2 and selection_marker_chunks is not None:
                # initialize reviewed code for applying code changes later a tonce
                reviewed_code = []

                # create line number stack for  merging code chunk with line numbers
                line_number_stack = []
                for line_number in reversed(review_json.keys()):
                    line_number_stack.append(utils.parse_string_to_int(line_number))

                code_chunks_to_review = []

                # prompt offset tokens
                prompt_tokens = utils.count_tokens(
                    json.dumps(
                        prompt.get_apply_review_for_file_prompt(
                            "",
                            "",
                            gpt_model.value,
                            programming_language,
                            gpt_model,
                        )
                    )
                )

                # iterate over code chunks by selection markers
                # and merge them with review suggestions by line numbers
                for code_chunk in selection_marker_chunks.values():
                    # if there are no more line numbers in stack,
                    # there are no more review suggestions, break loop
                    if not line_number_stack:
                        break

                    # merge code chunk with suggestions by line numbers
                    chunk_payload = formatter.parse_apply_review_per_code_hunk(
                        code_chunk,
                        review_json,
                        line_number_stack,
                    )

                    # there are review suggestions in that code chunk
                    if chunk_payload:
                        for chunk in chunk_payload:
                            chunk_tokens = (
                                utils.count_tokens(json.dumps(chunk)) + prompt_tokens
                            )
                            # if chunk tokens are smaller than threshold
                            # add chunk to code chunks to review
                            if chunk_tokens <= gpt_model.value / 2:
                                code_chunks_to_review.append(chunk)
                            else:
                                # code chunk tokens are greater than threshold
                                # skip since results are not reliable
                                pass

                if code_chunks_to_review:
                    code_chunk_count = code_chunks_to_review.__len__()
                    for index, chunk in enumerate(code_chunks_to_review, start=1):
                        reviewed_code_chunks = request_review_changes(
                            chunk,
                            api_key,
                            gpt_model,
                            programming_language,
                            index,
                            code_chunk_count,
                            file_name,
                        )
                        add_reviewed_code(reviewed_code_chunks, reviewed_code)

                file.close()
                code_lines: Dict[int, str] = formatter.code_block_to_dict(
                    "".join(reviewed_code)
                )
                utils.override_lines_in_file(absolute_file_path, code_lines)
                print(
                    "✅ Successfully applied review changes to "
                    f"{utils.get_bold_text(os.path.basename(absolute_file_path))}"
                    "\n"
                    "Note: The changes have been applied iteratively "
                    "due to the large amount of changes. "
                    "There might be syntax errors in the code. "
                    f"Consider using the {utils.get_bold_text('--gpt4')} flag."
                )

            # tokens for file content and review suggestions are less than threshold
            # send request for file content and review suggestions
            else:
                max_completions_tokens = gpt_model.value - tokens
                reviewed_git_diff = request.send_request(
                    api_key,
                    prompt.get_apply_review_for_file_prompt(
                        file_content,
                        json.dumps(payload["reviews"]),
                        max_completions_tokens,
                        programming_language,
                        gpt_model,
                    ),
                    f"🔧 Applying changes to {utils.get_bold_text(file_name)}...",
                )
                reviewed_git_diff = formatter.extract_content_from_markdown_code_block(
                    reviewed_git_diff
                )
                file.close()
                with open(absolute_file_path, "w") as file:
                    if reviewed_git_diff:
                        file.write(reviewed_git_diff)
                        print(
                            "✅ Successfully applied review changes to "
                            f"{utils.get_bold_text(file_name)}"
                        )

    except FileNotFoundError:
        print(f"💥 File '{absolute_file_path}' not found.")
    except IOError:
        print(f"💥 Error reading file '{absolute_file_path}'.")
    except ValueError as e:
        print(f"💥 Error while applying review changes for file {absolute_file_path}.")
        print(e)
    return None


def request_review_changes(
    code_chunk_with_suggestions,
    api_key,
    gpt_model,
    programming_language,
    current_step,
    total_steps,
    file_name,
):
    message_tokens = utils.count_tokens(
        json.dumps(
            prompt.get_apply_review_for_git_diff_chunk_promp(
                code_chunk_with_suggestions["code"],
                json.dumps(code_chunk_with_suggestions["suggestions"]),
                gpt_model.value,
                programming_language,
                gpt_model,
            )
        )
    )
    return request.send_request(
        api_key,
        prompt.get_apply_review_for_git_diff_chunk_promp(
            code_chunk_with_suggestions["code"],
            json.dumps(code_chunk_with_suggestions["suggestions"]),
            gpt_model.value - message_tokens,
            programming_language,
            gpt_model,
        ),
        "🔧 Applying changes to "
        + f"{utils.get_bold_text(file_name)}... {current_step}/{total_steps}",
    )


def add_reviewed_code(review_applied, reviewed_code):
    if review_applied:
        for (
            improved_code_block
        ) in formatter.extract_content_from_multiple_markdown_code_blocks(
            review_applied
        ):
            reviewed_code.append("\n" + improved_code_block)
