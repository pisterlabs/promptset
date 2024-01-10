from flask import *
from . import app_data_dir, logging, getExc, __version__, openai
from . import error_handler as exception_handler
from .imager import openai_handler
from os import path, environ
from .common import generator, history
from typing import Any

app_data_dir = path.join(app_data_dir, "contents")

generate = generator()


class local_config:
    def __init__(self):
        self.auth_cookie_key = "openai_api_key"
        self.upload_path = self.get_path("uploads")
        self.incomplete_form_msg = "Kindly fill all the fields"
        self.history = history(app_data_dir)
        self.history_args = ["category", "prompt", "time", "urls"]
        self.get_identity = lambda: request.cookies.get("id")

    def get_path(self, *args):
        return path.join(app_data_dir, "/".join(args))

    def get_arg(self, *keys, default: Any = None, resp: dict = {}) -> dict:
        """Extracts parameter's value from GET request

        Args:
            default (Any, optional): Value to be return incase of None. Defaults to None.
            resp (dict, optional): Keys,values to be added in response. Defaults to {}.

        Returns:
            dict: Parameter:Value
        """
        for arg in keys:
            resp[arg] = request.args.get(arg, default)
        return resp

    def get_from_form(self, *keys, resp: dict = {}) -> dict:
        """Exracts data from POST requests

        Args:
            resp (dict, optional): _description_. Defaults to {}.

        Returns:
            dict: Form data - key,value
        """
        for key in keys:
            resp[key] = request.form.get(key)
        return resp

    def history_handler(
        self, target: str = None, category: str = None, limit: int = 1000
    ) -> dict:
        """Extracts requested history data

        Args:
            target (str, optional): Data in need. Defaults to None.
            limit (int,optional) : Limit response[data]

        Returns:
            dict : Requested data
        """
        target = str(target).lower()
        if limit and str(limit).isdigit():
            limit = int(limit)
        else:
            limit = 0
        if str(category).lower() in ["bing", "masking", "chatgpt", "variation"]:
            category = category.lower()
        else:
            category = None
        resp = {"data": []}
        data = self.history.get_contents(self.get_identity())
        if target and target in self.history_args:
            for entry in data["data"]:
                current_resp = resp["data"]
                if category and entry["category"].lower() != category:
                    continue
                if target == "urls":
                    resp["data"] = current_resp + entry[target]
                else:
                    current_resp.append(entry[target])
                    resp["data"] = current_resp

        else:
            resp = data
            if category:
                harvested = []
                for entry in resp["data"]:
                    if entry["category"].lower() == category:
                        harvested.append(entry)
                resp = {"data": harvested}

        if limit and len(resp["data"]) > limit:
            resp["data"] = resp["data"][:limit]
        return resp

    def get_from_file(self, *keys, resp: dict = {}):
        """Retrieve files from form

        Args:
            resp (dict, optional): Dictionary to add the response. Defaults to {}.
            keys (str): Name of the files


        Returns:
            dict: name and filepath
        """
        for key in keys:
            file = request.files.get(key)
            if file:
                filepath = path.join(self.upload_path, file.filename)
                file.save(filepath)
                resp[key] = filepath
            else:
                resp[key] = None
        return resp

    def format_response(
        self,
        resp: list,
        http_code: int = 200,
        error: bool = None,
        prompt: str = "Image Variant",
        category="Variation",
    ):
        """Format response to be handled by API

        Args:
            resp (str | list): Response
            hhtp_code (int): Response http code
            error (bool, optional): Specifies to handle resp as error . Defaults to False.
            prompt (str, optional): Prompt parsed . Defaults to Image Variant
            category (str, optional): Image manipulator category

        Returns:
            json: Response formatted for API
        """
        if not isinstance(resp, list):
            error = True
            http_code = 501 if http_code == 200 else http_code
        api_data = {
            "url": resp if not error else None,
            "error": None if not error else resp,
        }
        if api_data["url"]:
            self.history.add_new(prompt, category, api_data["url"], self.get_identity())
        if not any(list(api_data.values())) or "an unexpected keyword" in str(
            api_data["error"]
        ):
            api_data[
                "error"
            ] = """
            <div style="text-align:center;color:lime;">Server seems to be misconfigured!<br>If you're the host, ensure the 
            <a href="https://github.com/acheong08/EdgeGPT#getting-authentication-required" target="_blank">bing cookie file</a>
            and <a href="https://platform.openai.com/account/api-keys" target="_blank">OpenAI API Key</a>
            are set properly.</div>"""
        return jsonify(api_data), http_code

    def imager_error_handler(self):
        """Exception handler at API level"""

        def decorator(func):
            def main(*args, **kwargs):
                try:
                    return func(*args, **args)
                except Exception as e:
                    return jsonify({"error": self.format_response(getExc(e))})

            return main

        return decorator


def API(
    args: object,
    port: int = 8000,
    debug: bool = True,
    host: bool = False,
    threaded: bool = False,
):
    """Start the web app

    Args:
        args (object) : Argparse object.
        port (int, optional): Port for web to listen. Defaults to 8000.
        debug (bool, optional): Start the app in debug mode. Defaults to True.
        host (bool | str, optional): Host the web on LAN. Defaults to False.

    Returns:
        None: None
    """
    api_config = local_config()
    openai = openai_handler(args)

    app = Flask(
        __name__,
        static_folder=api_config.get_path("static"),
        template_folder=api_config.get_path("templates"),
    )

    @api_config.imager_error_handler()
    @app.route("/")
    def index():
        """Landing page"""
        resp = make_response(render_template("index.html"))
        if not api_config.get_identity():
            resp.set_cookie("id", generate.new_cookie(), 259200)
        return resp

    @api_config.imager_error_handler()
    @app.route("/v1/history", methods=["GET"])
    def image_history():
        # args : amount  category
        data = api_config.history.get_contents(api_config.get_identity())
        reqs = api_config.get_arg("limit", "target", "category")
        return api_config.history_handler(**reqs)

    @api_config.imager_error_handler()
    @app.route("/v1/image/<action>", methods=["GET"])
    def imager(action):
        """Handle v1 routings"""
        if not action in ("prompt", "variation", "mask", "bing"):
            action = "prompt"
        return render_template(
            "form.html", category=action, action=f"/v1/image/{action}/generate"
        )

    @api_config.imager_error_handler()
    @app.route("/v1/image/prompt/generate", methods=["POST"])
    def create_from_prompt():
        """Generate image from text"""
        form_data = api_config.get_from_form("prompt", "total_images", "image_size")
        if all(list(form_data.values())):
            resp = openai.create_from_prompt(**form_data)
            return api_config.format_response(
                resp, prompt=form_data["prompt"], category="ChatGPT"
            )
        else:
            return api_config.format_response(
                api_config.incomplete_form_msg, http_code=400
            )

    @api_config.imager_error_handler()
    @app.route("/v1/image/bing/generate", methods=["POST"])
    def create_with_bing():
        """Generate image with bing"""
        form_data = api_config.get_from_form("prompt", "total_images")
        if all(list(form_data.values())):
            resp = openai.create_with_bing(**form_data)
            return api_config.format_response(
                resp, prompt=form_data["prompt"], category="Bing"
            )
        else:
            return api_config.format_response(
                api_config.incomplete_form_msg, http_code=400
            )

    @api_config.imager_error_handler()
    @app.route("/v1/image/mask/generate", methods=["POST"])
    def edit_with_mask():
        """Edit image with mask"""
        files = api_config.get_from_file("original_image_path", "masked_image_path")
        texts = api_config.get_from_form("prompt", "total_images", "image_size")
        files.update(texts)
        if all(list(files.values())):
            resp = openai.create_edit(**files)
            return api_config.format_response(
                resp, prompt=files["prompt"], category="Masking"
            )
        else:
            return api_config.format_response(
                api_config.incomplete_form_msg, http_code=400
            )

    @api_config.imager_error_handler()
    @app.route("/v1/image/variation/generate", methods=["POST"])
    def get_variation():
        """Get another image like same"""
        files = api_config.get_from_file("path_to_image")
        texts = api_config.get_from_form("total_images", "image_size")
        files.update(texts)
        if all(list(files.values())):
            resp = openai.create_variation(**files)
            return api_config.format_response(resp)
        else:
            return api_config.format_response(
                api_config.incomplete_form_msg, http_code=400
            )

    launch_configs = {
        "port": port,
        "debug": debug,
        "threaded": threaded,
    }
    if host:
        launch_configs["host"] = "0.0.0.0"

    app.run(**launch_configs)


@exception_handler(log=False)
def start_server():
    """Server entry"""
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Manipulate images with OpenAI's model",
        epilog="This script has no official relation with OpenAI.",
    )
    parser.add_argument(
        "port", nargs="?", type=int, help="Port to start the server at", default=8000
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s v{__version__}"
    )
    parser.add_argument("-k", "--key", help="OpenAI's API key")
    parser.add_argument(
        "-kp", "--key-path", help="Path to OpenAI-API-KEY path", metavar="PATH"
    )
    parser.add_argument(
        "-l",
        "--logging-level",
        type=int,
        help="Log level of the app",
        choices=[10, 20, 30, 40, 50],
        metavar="10-50",
        default=20,
    )
    parser.add_argument("-o", "--output", help="Filepath to log to", metavar="PATH")
    parser.add_argument(
        "-cf", "--cookie-file", metavar="PATH", help="Path to Bing's cookie file"
    )
    parser.add_argument("--host", action="store_true", help="Host the site on LAN")
    parser.add_argument(
        "--thread", help="Run server in multiple threads", action="store_true"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Start as debugging server"
    )
    args = parser.parse_args()
    if args.key:
        openai.api_key = args.key

    if args.key_path:
        openai.api_key_path = args.key_path

    if any([args.logging_level, args.output]):
        log_config = {
            "format": "[%(asctime)s] - %(levelname)s : %(message)s",  # %(module)s:%(lineno)s",
            "datefmt": "%H:%M:%S %d-%b-%Y",
            "level": args.logging_level,
        }
        if args.output:
            log_config["filename"] = args.output
        logging.basicConfig(**log_config)
    if not any([environ.get("OPENAI_API_KEY"), openai.api_key, openai.api_key_path]):
        exit(logging.critical("OpenAI-API-Key is required"))
    API(args, args.port, args.debug, args.host, args.thread)
