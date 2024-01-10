import uuid
import openai
import requests
from api_gpt.services.openai_request import *
from api_gpt.nlp.v1.generation import *
from api_gpt.nlp.classify import is_single_task
from api_gpt.nlp.exploration_simple import simple_task_exploration
from api_gpt.nlp.generation import *
from api_gpt.nlp.generation import generate_workflow
from api_gpt.utils import *
from api_gpt.workflows.db.apps_template import (
    dump_template_from_generated_workflow,
    write_apps_template,
)
from api_gpt.workflows.execute.execute import execute_intent, execute_intent_from_json
from google.protobuf.json_format import MessageToJson
from timeit import default_timer as timer
from api_gpt.settings.debug import global_debug_flag
from firebase_admin import credentials, db


@current_app.route("/workflow/generate_exploration", methods=["POST"])
@token_required
def workflow_generate_exploration():
    try:
        _json = request.json
        model = _json["model"]
        text = get_key_or_none(_json, "text")
        max_tokens = get_key_or_default(_json, "max_tokens", 500)

        start = timer()
        is_simple_task = is_single_task(text, model, max_tokens)
        end = timer()
        seconds = int((end - start) * 100) / 100
        print(f"Classification took {seconds} seconds", flush=True)

        if is_simple_task:
            start = timer()
            workflow, openai_response = simple_task_exploration(text, model, max_tokens)
            end = timer()
            seconds = int(end - start)
            print(f"single step generartion took {seconds} seconds", flush=True)
        else:
            start = timer()
            workflow, openai_response = generate_workflow_by_exploration(
                text, model, max_tokens
            )
            end = timer()
            seconds = int(end - start)
            print(f"multi step generartion took {seconds} seconds", flush=True)

        if workflow != None:
            workflow_str = MessageToJson(workflow)
            return json.loads(workflow_str)
        else:
            return openai_response, 502

    except Exception as e:
        return "Failed with : " + str(e), 500


@current_app.route("/workflow/generate_form", methods=["POST"])
@token_required
def workflow_generate_form():
    try:
        model = request.form.get("model", "")
        text = request.form.get("text", "")
        max_tokens = request.form.get("max_tokens", 500)
        use_template = request.form.get("use_template", False)
        exploration = request.form.get("exploration", False)

        workflow, error_message, error_code = generate_workflow(
            text, model, max_tokens, use_template, use_exploration=exploration
        )

        if workflow != None:
            workflow_str = MessageToJson(workflow)
            return json.loads(workflow_str)
        else:
            return error_message, error_code

    except Exception as e:
        return "Failed with : " + str(e), 500


@current_app.route("/workflow/generate", methods=["POST"])
@cross_origin()
@token_required
def workflow_generate():
    try:
        _json = request.json
        model = _json["model"]
        text = get_key_or_none(_json, "text")
        max_tokens = get_key_or_default(_json, "max_tokens", 500)
        use_template = get_key_or_default(_json, "use_template", False)
        exploration = get_key_or_default(_json, "exploration", True)
        user_context = get_key_or_default(_json, "user_context", "")
        workflow_id = get_key_or_default(_json, "workflow_id", str(uuid.uuid4()))
        if global_debug_flag:
            print(
                f"generate flags : max_tokens:{max_tokens}, text:{text}, use_template:{use_template}, exploration:{exploration}"
            )

        start = timer()

        workflow, error_message, error_code = generate_workflow(
            text,
            model,
            max_tokens,
            use_template,
            use_exploration=exploration,
            user_context=user_context,
        )

        end = timer()
        steps = 0
        if workflow != None:
            dump_template_from_generated_workflow(workflow)
            steps = len(workflow.intent_data)
        seconds = end - start
        print(
            f"elaped time for generate model with {steps} steps, elaped {int(seconds)} seconds.",
            flush=True,
        )

        if workflow != None:
            workflow.id = workflow_id
            workflow_str = MessageToJson(workflow)
            workflow_json = json.loads(workflow_str)
            try:
                workflow_ref = db.reference(f"plasma/workflows/{workflow.id}/workflow")
                workflow_ref.set(workflow_json)
            except Exception as e:
                pass
            return workflow_json
        else:
            return error_message, error_code

    except Exception as e:
        print(f"failed at workflow generate : {e}")
        return "Failed with : " + str(e), 500


@current_app.route("/workflow_generation_v3", methods=["POST"])
@token_required
def plasma_ai_generation_v3():
    try:
        _json = request.json
        model = _json["model"]
        text = get_key_or_none(_json, "text")
        max_tokens = get_key_or_default(_json, "max_tokens", 3000)
        is_success, workflow = generate_ai_generation_v3(text=text, model=model)
        if is_success:
            return json.loads(MessageToJson(workflow))
        else:
            return "Failed in workflow generation", 501

    except Exception as e:
        return "Failed with : " + str(e), 500


@current_app.route("/workflow/execute", methods=["POST"])
@token_required
def workflow_execute():
    try:
        _json = request.json
        intent_data_json = get_key_or_none(_json, "intent_data")
        execution_data, error_message, error_code = execute_intent_from_json(
            intent_data_json
        )
        if execution_data == None:
            return error_message, error_code
        else:
            return json.loads(MessageToJson(execution_data))

    except Exception as e:
        return "Failed with : " + str(e), 500
