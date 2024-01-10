import os.path

from chaincrafter import Chain, Catalog
from chaincrafter.models import OpenAiChat

chat_model = OpenAiChat(
    temperature=0.9,
    model_name="gpt-3.5-turbo",
    presence_penalty=0.1,
    frequency_penalty=0.2,
)

# Load the catalog
catalog = Catalog()
path = os.path.dirname(__file__)
catalog.load(os.path.join(path, "catalog.yml"))

# Use the catalog to get the prompts and build a chain
chain = Chain(
    catalog["worldly_helpful_assistant"],
    (catalog["hello_france"], "city"),
    (catalog["city_population"], "followup_response"),
)

# Use the catalog to get an already assembled chain
chain = catalog.get_chain("hello")

# Run the chain
messages = chain.run(chat_model)
for message in messages:
    print(f"{message['role']}: {message['content']}")
