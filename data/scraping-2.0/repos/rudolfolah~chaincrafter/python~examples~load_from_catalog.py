import getopt
import os
from string import Formatter
import sys

from chaincrafter.catalogs import Catalog
from chaincrafter.models import OpenAiChat

opts, args = getopt.getopt(
    sys.argv[1:],
    "lr",
    ["list", "run="]
)

if len(opts) == 0:
    print("""Usage:
\tpython load_from_catalog.py --list
\tpython load_from_catalog.py --run <chain_name>
""")
    sys.exit(1)

catalog = Catalog()
path = os.path.dirname(__file__)
catalog.load(os.path.join(path, "catalog.yml"))

for opt, arg in opts:
    if opt in ("-l", "--list"):
        print("Available chains:")
        for chain_name in catalog.chains:
            print(f"{chain_name}")
        sys.exit(0)

    if opt in ("-r", "--run"):
        if arg is None or arg == "":
            print("Chain name must be provided")
            sys.exit(1)
        if arg not in catalog.chains:
            print(f"Chain {arg} not found")
            sys.exit(1)
        chat_model = OpenAiChat(
            temperature=0.9,
            model_name="gpt-3.5-turbo",
            presence_penalty=0.1,
            frequency_penalty=0.2,
        )
        chain_name = arg
        chain = catalog.get_chain(chain_name)
        fmt_str = chain._prompts_and_output_keys[0][0]._fmt_str
        formatter = Formatter()
        starting_input_vars = {
            var_name: input(f"Enter value for {var_name}: ")
            for _, var_name, _, _ in formatter.parse(fmt_str)
            if var_name is not None
        }
        messages = chain.run(chat_model, starting_input_vars)
        for message in messages:
            print(f"{message['role']}: {message['content']}")
        sys.exit(0)
