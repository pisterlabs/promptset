from ai4teaching import Assistant
from ai4teaching import log
from openai import OpenAI

class RetrievalAssistant(Assistant):

    def __init__(self, config_file, depending_on_assistant=None):
        log("Initializing RetrievalAssistant", type="debug")
        log(f"RetrievalAssistant depends on {depending_on_assistant}", type="debug") if depending_on_assistant else None
        super().__init__(config_file, depending_on_assistant)

        self.openai_client = OpenAI()
        self.openai_model = "gpt-4-1106-preview"
        
        self.last_prompt = []
        self.messages = []
        self.system_message = None

    def chat(self, message, model="gpt-4-1106-preview"):
        
        # Create new message entry from text and add
        new_message_json = {"role": "user", "content": f"{message}"}

        condensed_question = self._condense_messages_for_retrieval(message, self.messages)

        # Save for debug
        self.last_condensed_question = condensed_question

        # Add the message from the user to the messages
        self.messages.append(new_message_json)

        # Retrieve documents for condensed question
        retrieved_documents = self.vector_db.query(condensed_question, n_results=5)

        retrieved_documents_string = ""
        for doc in retrieved_documents["documents"][0]:
            retrieved_documents_string += f"\"{doc}\"\n\n"

        # Save for debug
        self.last_retrieved_documents = retrieved_documents
        
        # Create prompt with retrieved documents
        prompt = f"""Beantworte die folgende Frage eines Bürgers ausschließlich auf Basis der folgenden Texte, die aus Dokumenten zum Food Future Lab stammen. Erfinde keine Antworten, wenn die notwendigen Informationen nicht in den Texten enthalten sind. Sag dann einfach, dass du darüber keine Informationen vorliegen hast. Leite deine Antwort mit folgenden oder ähnlichen Wörten ein: \"Basierend auf meinem Wissen...\", \"Meines Wissens nach...\", \"Soweit mir bekannt...\". Erwähne NIEMALS, dass dir Dokumente oder Textauszüge vorliegen. Vermeide also Sätze wie "in den vorligenden Dokumenten" oder "in den mir bekannten Texten..." etc.
        
        Frage eines Bürger: \"{condensed_question}\"

        Texte aus Dokumenten:

        {retrieved_documents_string}
        """

        prompt = '\n'.join([m.lstrip() for m in prompt.split('\n')])

        self.last_prompt = prompt

        # Complete the prompt
        response = self.openai_client.chat.completions.create(
            model=model,
            messages= [ { "role": "user", "content": prompt } ]
        )

        response_message = {"role": "assistant", "content": response.choices[0].message.content}

        # Add the assistants response to the messages
        self.messages.append(response_message)
        
        return self._get_cleaned_messages_copy()
    
    def _condense_messages_for_retrieval(self, current_prompt, messages):

        chat_history = ""
        for message in messages:
            chat_history += f"{message['role']}: {message['content']}\n\n"

        condense_prompt = f"""Formuliere eine eigenständige Frage auf Basis des folgenden Chatverlaufs sowie der letzte Benutzerfrage, die sich auf den Chatverlauf beziehen könnte. Die eigenständige Frage soll ohne den Chatverlauf verstanden werden kann. Beantworte die Frage NICHT, sondern formuliere sie nur um, wenn es nötig ist, und gib sie ansonsten so zurück, wie sie ist. Auf Basis dieser neuen Frage wird eine semantische Suche durchgeführt, um die Antwort zu ermitteln.

        Chatverlauf:

        {chat_history}
        Letzte Benutzerfrage: \"{current_prompt}\"

        Eigenständige Frage: """

        condense_prompt = '\n'.join([m.lstrip() for m in condense_prompt.split('\n')])

        #log(f"{condense_prompt}", type="debug")

        condense_messages = []
        condense_instruction_message = {"role": "user", "content": f"{condense_prompt}"}
        condense_messages.append(condense_instruction_message)

        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=condense_messages
        )

        log(f"Reformulated question: {response.choices[0].message.content}", type="debug")

        return response.choices[0].message.content


    def load_system_message(self, system_message_file):
        with open(system_message_file, "r", encoding="utf-8") as f:
            system_message = f.read()

        self.set_system_message(system_message)
    
    def set_system_message(self, system_message):
        self.system_message = system_message

        # Replace or add system message
        for m in self.messages:
            if m["role"] == "system":
                m["content"] = system_message
                return
        
        # If we are here, there way no system message, add to first position
        self.messages.insert(0, {"role": "system", "content": system_message})

    def reset(self):
        log("Resetting RetrievalAssistant", type="debug")
        self.messages = []

        # Add system message if present
        if self.system_message is not None:
            self.messages.append({"role": "system", "content": self.system_message})

        return self._get_cleaned_messages_copy()
    
    def _get_cleaned_messages_copy(self):
        # Remove system message
        messages_copy = self.messages.copy()
        if self.system_message is not None:
            messages_copy.pop(0)
        return messages_copy
