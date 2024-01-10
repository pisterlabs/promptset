import textwrap
import shutil
import typer
import langchain
from langchain.base_language import BaseLanguageModel
from langchain.llms.openai import OpenAI
from langchain.agents.agent import Agent
from lmi.app import App

description = """\
                                               
     ▄            ▄▄       ▄▄  ▄▄▄▄▄▄▄▄▄▄▄     
    ▐░▌          ▐░░▌     ▐░░▌▐░░░░░░░░░░░▌    
    ▐░▌          ▐░▌░▌   ▐░▐░▌ ▀▀▀▀█░█▀▀▀▀     
    ▐░▌          ▐░▌▐░▌ ▐░▌▐░▌     ▐░▌         
    ▐░▌          ▐░▌ ▐░▐░▌ ▐░▌     ▐░▌         
    ▐░▌          ▐░▌  ▐░▌  ▐░▌     ▐░▌         
    ▐░▌          ▐░▌   ▀   ▐░▌     ▐░▌         
    ▐░▌          ▐░▌       ▐░▌     ▐░▌         
    ▐░█▄▄▄▄▄▄▄▄▄ ▐░▌       ▐░▌ ▄▄▄▄█░█▄▄▄▄     
    ▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌    
     ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀     
                                               
     —————————————————————————————————————     
          LANGUAGE  MODEL  INTERFACE           
                                               
      Copyright (c) 2023 by ComputaCo Inc.
      Released under the MIT License (MIT)
                                               
     —————————————————————————————————————     
"""
disclaimer = """THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""

cli_app = typer.Typer(name="LMI", help=description)


def load_app(path: str) -> App:
    if ":" in path:
        path, app_object_name = path.split(":")
    else:
        app_object_name = None
    module = __import__(path)
    if not app_object_name:
        return next(
            module.__dict__[name]
            for name in module.__dict__
            if isinstance(module.__dict__[name], App)
        )
    else:
        if not app_object_name in module.__dict__:
            raise ValueError(f"Object {app_object_name} not found in module {path}")
        app = module.__dict__[app_object_name]
        if not isinstance(app, App):
            raise ValueError(f"Object {app_object_name} in module {path} is not an App")
        return app


def load_llm(llm: str, llm_config={}) -> BaseLanguageModel:
    # FIXME: find more principled way to do this
    match llm.lower().strip():
        case "openai":
            llm = OpenAI(model_name="gpt-4-1106", **llm_config)
        case r"openai:([a-zA-Z0-9\-_]+)":
            llm = OpenAI(model_name=llm.split(":")[1], **llm_config)
        case _:
            raise ValueError(f"LLM {llm} not found")
    return llm


@cli_app.command()
def serve(
    app: str = typer.Argument(
        help='Path to the app to serve, optionally with a ":app_object_name" suffix to specify the app object to serve'
    ),
    port: int = typer.Option(..., help="The port to serve on"),
):
    typer.echo(description)

    app: App = load_app(app)
    app.serve(port=port)

    typer.echo("Serving...")


@cli_app.command()
def cli():
    typer.echo(description)

    app: App = load_app(app)
    app.cli()

    typer.echo("Serving...")


@cli_app.command()
def run(
    app: str = typer.Argument(
        ...,
        help='Path to the app to run, optionally with a ":app_object_name" suffix to specify the app object to run',
    ),
    agent: str = typer.Option(
        ..., help="Serialized langchain agent to run the app against"
    ),
    llm: str = typer.Argument(..., help="LLM to run the app against"),
):
    typer.echo(description)

    app: App = load_app(app)
    llm = load_llm(llm)
    raise NotImplementedError("TODO: load agent from the lc hub")

    app.run(agent=agent)

    typer.echo("Running...")


@cli_app.callback()
def main():
    typer.echo(description)

    console_width = shutil.get_terminal_size()[0]
    print("\n".join(textwrap.wrap(disclaimer, width=console_width)))

    typer.echo("\nPlease specify a subcommand.")
    typer.Exit(0)


if __name__ == "__main__":
    cli_app()
