from openai import OpenAI
from config import config
from datetime import datetime, timezone

client = OpenAI(
    api_key=config.openai_api_key,
    organization=config.openai_organization_id
)

if __name__ == "__main__":
    # Assuming you have the cost per 1k tokens
    cost_per_1k_tokens = 0.0080
    cost_per_token = cost_per_1k_tokens / 1000

    # list all the models fine-tuned.
    result = client.fine_tuning.jobs.list(limit=10)
    result_list = []
    for item in result:
        created_at_utc = datetime.fromtimestamp(item.created_at, tz=timezone.utc)
        created_at_formatted = created_at_utc.strftime('%Y-%m-%d %H:%M:%S')
        error_message = item.error.message if item.error else None
        model_cost = None
        if item.trained_tokens is not None:
            model_cost = item.trained_tokens * cost_per_token
        fine_tuned_model_info = {
            "model_name": item.fine_tuned_model,
            "status": item.status,
            "trained_tokens": item.trained_tokens,
            "model_cost": model_cost,
            "error": error_message,
            "created_at": created_at_formatted,
        }
        result_list.append(fine_tuned_model_info)

    # print the result
    for model_info in result_list:
        print(f"Fine-tuning model >>> {model_info}")
