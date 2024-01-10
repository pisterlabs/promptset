from yowsup.layers import YowParallelLayer, YowLayerEvent
from yowsup.layers.auth import YowAuthenticationProtocolLayer
from yowsup.layers.network import YowNetworkLayer
from yowsup.layers.protocol_messages import YowMessagesProtocolLayer
from yowsup.layers.protocol_receipts import YowReceiptProtocolLayer
from yowsup.layers.protocol_presence import YowPresenceProtocolLayer
from yowsup.layers.protocol_iq import YowIqProtocolLayer
from yowsup.layers.logger import YowLoggerLayer
import openai

openai.api_key = "YOUR_API_KEY"
class MessageListenerLayer(YowParallelLayer):
    def __init__(self):
        super(MessageListenerLayer, self).__init__()
        self.ackQueue = []
        openai.api_key = "YOUR_API_KEY"

    def receive(self, event):
        if event.name == "message":
            self.onMessageEvent(event)

    def onMessageEvent(self, event):
        node = event.getFrom(False)
        if event.getType() == "text":
            incoming_message = event.getBody().lower().strip()
            intent = understand_intent(incoming_message)
            prompt = f"I want to {intent}. {incoming_message}"
            response = generate_response(prompt)
            outgoing_message = response.replace("\n\n", " ")

            outgoing_event = YowLayerEvent(
                YowMessagesProtocolLayer.EVENT_SEND_MESSAGE,
                (node, outgoing_message),
            )
            self.ackQueue.append(outgoing_event)
            self.broadcastEvent(outgoing_event)

    def generate_response(self, prompt):
        prompt = f"Desire: {prompt}"
        response = openai.Completion.create(
            prompt=prompt,
            temperature=0.7,
            max_tokens=100,
            n=1,
            stop=None,
        )
        generated_text = response.choices[0].text
        generated_text += "\n\nThe Alien TV team\n(Powered by TEXT GENIUS)"
        return generated_text

    def understand_intent(self, message):
        stop_words = stopwords.words("english")

        # Preprocessing
        words = message.lower().split()
        words = [word for word in words if word not in stop_words]
        message = " ".join(words)

        # Extract sentiment
        sentiment = TextBlob(message).sentiment.polarity

        # Return intent based on sentiment
        if sentiment > 0:
            return "praise"
        elif sentiment < 0:
            return "complain"
        else
            return "request"

if __name__ == "__main__":
    stack = MessageListenerLayer()
    stack.start()
