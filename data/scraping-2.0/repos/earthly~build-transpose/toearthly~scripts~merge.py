import argparse
import traceback
from textwrap import dedent

import openai

from toearthly.core import boot, constants, io  # noqa: F401
from toearthly.prompt import merge

# Default directories
DEFAULT_INPUT_DIR = "/input/"
DEFAULT_EARTHFILE_PATH = "/input/Earthfile"
DEFAULT_DEBUG_DIR = "/input/.to_earthly/"

def main(input_dir: str, earthfile_path: str) -> None:
    io.log("Starting")
    try:
        file1 = io.relative_read("data/merge/in1a.Earthfile")
        file2 = io.relative_read("data/merge/in1b.Earthfile")
        print(
            dedent(
              f"""
              Input:
              Earthfile1:\t{file1}
              Earthfile2:\t{file2}
              Output:\t\t{earthfile_path}
              Debug files:\t{constants.DEBUG_DIR}
              """
            )
        )

        print("Starting...\n (This may take 10 minutes)")
        print("Running Stage 1 - Dockerfile To Earthfile")
        earthfile = merge.prompt(file1, "python.yml", file2, "Dockerfile")
        io.verify(earthfile)
        io.write(constants.EARTHLY_WARNING + earthfile, earthfile_path)
    except openai.error.InvalidRequestError as e:
        print("Error: We were unable to convert this workflow.")
        io.log(f"Error Type: openai.error.InvalidRequestError \n Error details: {e}")
    except (ValueError, TypeError, IndexError, KeyError) as e:
        print("Error: We were unable to convert this workflow.")
        trace = traceback.format_exc()
        io.log(f"Error Type: {type(e).__name__} \n Error details: {e}")
        io.log(f"Stack Trace: {trace}")


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", help="Base file location", default=DEFAULT_INPUT_DIR
    )
    parser.add_argument(
        "--earthfile", help="Earthfile path", default=DEFAULT_EARTHFILE_PATH
    )
    parser.add_argument(
        "--debug_dir", help="Debug directory location", default=DEFAULT_DEBUG_DIR
    )
    return parser

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    constants.DEBUG_DIR = args.debug_dir
    print(f"${args.input_dir} {args.earthfile}")
    main(args.input_dir, args.earthfile)
