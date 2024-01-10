from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from langchain.document_loaders.base import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.utilities import ApifyWrapper
import os

os.environ["OPENAI_API_KEY"] = "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"
os.environ["APIFY_API_TOKEN"] = "apify_api_F0dQ71Q5UMGCj6FpKgNk7FnD6BDgCF1TZTZv"

apify = ApifyWrapper()

loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "https://www.vitoria.es.gov.br/perguntas_respostas.php"}]},
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"] or "", metadata={"source": item["url"]}
    ),
)

index = VectorstoreIndexCreator().from_loaders([loader])

app = Flask(__name__)

@app.route("/whatsapp", methods=['POST'])
def whatsapp_reply():
    """Responde a mensagens de WhatsApp entrantes."""
    incoming_message = request.values.get('Body', '').lower()

    print(incoming_message)

    # result = index.query_with_sources(query)

    resp = MessagingResponse()
    msg = resp.message()
    msg.body(str(incoming_message))

    return str(resp)

if __name__ == "main":
    app.run(debug=True)
