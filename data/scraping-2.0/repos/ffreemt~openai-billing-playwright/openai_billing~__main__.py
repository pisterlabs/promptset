"""Prep __main__.py."""
# pylint: disable=invalid-name,too-many-statements,too-many-branches, too-many-locals, too-many-arguments
import re
from pathlib import Path
from signal import SIG_DFL, SIGINT, signal
from typing import List, Optional
from urllib.parse import urlparse

import charset_normalizer
import pyperclip
import rich
import typer
from loguru import logger
from playwright.sync_api import sync_playwright
from typing_extensions import Annotated

from openai_billing import __version__
from openai_billing.get_browser import get_browser
from openai_billing.openai_billing import login_page, openai_billing

signal(SIGINT, SIG_DFL)

app = typer.Typer(
    name="openai-billing",
    add_completion=False,
    help="Check openai credit balance and expiration date",
)


def _version_callback(value: bool) -> None:
    logger.trace(f"{value=}")
    if value:
        typer.echo(f"{app.info.name} v.{__version__}")
        raise typer.Exit()


def parse_input(text: str) -> List[List[str]]:
    """Parse to obtain email/password pairs."""
    text = " ".join(text.splitlines())
    patt = re.compile(r"[\w+]+@\w+(?:\.\w+)+[, ]+\S+")
    return [re.split(r"[, ]+", elm, 1) for elm in re.findall(patt, text)]


@app.command()
def main(
    email_pw_pairs: Annotated[
        Optional[List[str]],
        typer.Argument(
            help=(
                "email-password pairs delimetered by space or ',', e.g. "
                "u1@gmail.com pw1 other stuff u2@gmail.com pw2 or u1@gmail.com, pw1 other stuff "
                " u2@gmail.com pw2 etc. Since passwords "
                "may contain special characters that may be interpreted by the shell, it's best to "
                """enclode the input with ", e.g. "u1@gmail.com pw1 other stuff u2@gmail.com pw2*&a", """
            ),
            show_default=False,
        ),
    ] = None,
    filepath: Annotated[
        Optional[str],
        typer.Option(
            "--filename",
            help="Filename/path that contains email/password pairs.",
        ),
    ] = None,
    clipb: Annotated[
        Optional[bool],
        typer.Option(
            "--clipb",
            "-c",
            help="Use clipboard content if set or if `email_pw_pairs` is empty.",
        ),
    ] = None,
    proxy: Annotated[
        Optional[str],
        typer.Option(
            "--proxy",
            help="Set a proxy, e.g., socks5://... or https://... etc",
        ),
    ] = None,
    headful: Annotated[
        bool,
        typer.Option(
            "--headful",
            help="Whether to go headless mode, i.e., --headful will display a browser.",
        ),
    ] = False,
    version: Annotated[  # pylint: disable=unused-argument
        Optional[bool],
        typer.Option(  # pylint: disable=unused-argument
            "--version",
            "-v",
            "-V",
            help="Show version info and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
):
    """Check openai credit balance and expiration date."""
    headless = not headful
    if proxy is not None:
        logger.trace(f"{proxy=}")
        if not all(urlparse(proxy)[:2]):
            rich.print(
                f"[green] {proxy=} [bold][red] does not seem to be a valid proxy. [/red][/bold] "
                "(Note a protocol, e.g. socks5, http, https etc., must be specified) "
                "We proceed nevertheless."
            )
    logger.trace(f"{headless=}")

    logger.trace(" parsing input for email/password pairs")
    if filepath:
        try:
            _ = Path(rf"{filepath}").read_bytes()
        except Exception as exc:
            logger.error(exc)
            raise SystemExit(1) from exc
        encoding = charset_normalizer.detect(_).get("encoding")
        if encoding is None:
            rich.print(f"{filepath=} is likely not a text file.")
            raise SystemExit(1)
        # else:
        try:
            text = Path(filepath).read_text(encoding=encoding)  # type: ignore
        except Exception as exc:
            logger.error(exc)
            raise SystemExit(1) from exc
        pairs = parse_input(text)
    else:
        if clipb:
            pairs = parse_input(pyperclip.paste())
        else:
            if email_pw_pairs:
                logger.trace(f"{email_pw_pairs=}")
                pairs = parse_input(" ".join(email_pw_pairs))
            else:
                pairs = parse_input(pyperclip.paste())
    if not pairs:
        rich.print("No email/password pairs found")
        raise typer.Exit()

    # logger.trace(f"to be checked: {pairs}")
    logger.debug(f"to be checked: {[elm[0] for elm in  pairs]}")

    # prepare a browser
    logger.trace("get a playwright loop")
    _ = """
    try:
        playwright = sync_playwright().start()
    except Exception as exc:
        logger.error(exc)
        raise SystemExit(1) from exc
    # """

    with sync_playwright() as playwright:
        logger.trace("get a browser")
        try:
            browser = get_browser(playwright, proxy=proxy, headless=headless)
        except Exception as exc:
            logger.error(exc)
            raise SystemExit(1) from exc

        logger.trace("login to openai and fetch info")
        rich.print("[green][bold]\tlogin to openai and fetch info...")
        for elm in pairs:
            ctx = browser.new_context()
            try:
                page = login_page(elm[0], elm[1], ctx)
                res = openai_billing(page)
                rich.print(f"{elm[0]}, [green]{res}")
            except Exception as exc:
                res = str(exc)
                rich.print(f"{elm[0]}, [green]{res}")


if __name__ == "__main__":
    app()
