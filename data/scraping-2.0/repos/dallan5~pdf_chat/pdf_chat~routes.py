import os
import openai
from flask import request, render_template, jsonify, views, session, url_for, send_from_directory
from werkzeug.utils import secure_filename

from .utils import markdown_to_html, format_message, extract_text_from_page, clear_messages, setup_messages, clear_directory
from .session import get_session_data, set_session_data, update_session_messages
from .config import Config

def register_message(role, content):
    
    conversation_messages = get_session_data("conversation_messages", [])
    conversation_messages.append(format_message(role, content))
    set_session_data("conversation_messages", conversation_messages)
    update_session_messages()

def extract_first_page_text(pdf_path):
    return extract_text_from_page(pdf_path, 0)

class ChatView(views.MethodView):
    
    def get(self):
        # Set default values if nothing provided
        if 'conversation_messages' not in session:
            setup_messages()
        if "source_page" not in session:
            set_session_data("source_page", 0)
        if "pdf_path" not in session:
            set_session_data("pdf_path", "static/pdf/book.pdf")
        if "system_messages" not in session:
            set_session_data("system_messages", [])
        if "source_text" not in session:
            set_session_data("source_text", extract_first_page_text(get_session_data("pdf_path")))
        update_session_messages()

        return render_template("index.html", messages=get_session_data("conversation_messages", []))

    def post(self):
        if request.is_json:
            return self._handle_ajax_request()

    def _handle_ajax_request(self):
        query = request.json['query']
        register_message("user", query)  # Keep this line
        # Flag to control the while loop
        retry_request = True
        # Counter to keep track of the number of attempts
        attempt_counter = 0
        # Maximum number of attempts
        max_attempts = 2

        content = ""
        role = "assistant"
        while retry_request and attempt_counter < max_attempts:
            attempt_counter += 1  # Increment the counter at the beginning of each loop iteration

            try:
                # Your OpenAI API request
                print("sending request")
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=get_session_data("system_messages", []),
                    temperature=0,
                    max_tokens=1024,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                print("response")
                # If the request is successful, the following lines will execute,
                # and the loop will exit by setting retry_request to False
                role = response['choices'][0]['message']['role']
                content = markdown_to_html(response['choices'][0]['message']['content'])
                register_message(role, content)
                retry_request = False  # Set the flag to false to exit the loop

            except openai.error.RateLimitError as e:
                print(f"OpenAI API request exceeded rate limit: {e}")
                # Optionally, you may want to break out of the loop or handle this error differently

            except openai.error.OpenAIError as e:
                # Assuming the token error is of type OpenAIError, adjust if necessary
                print(f"Token error: {e}. Clearing messages and retrying...")
                set_session_data("system_messages", clear_messages(get_session_data("system_messages", [])))

            except Exception as e:
                print(f"Error: {e}")
                content = "Sorry, I couldn't process that request."
                register_message("assistant", content)
                retry_request = False  # Set the flag to false to exit the loop

        # Check if the loop exited due to reaching the maximum number of attempts
        if attempt_counter >= max_attempts:
            content = "Sorry, I couldn't process that request after multiple attempts."
            register_message("assistant", content)

        # Return the response to the frontend
        # This is placed outside the loop to ensure it's executed once the loop exits
        return jsonify({"role": role, "message": content})
    

class MemoryView(views.MethodView):

    def post(self):
        print(request.path)
        if request.path.endswith('clear_session'):
            return self.clear_session()
        elif request.path.endswith('capture_text'):
            return self.capture_text()
        elif request.path.endswith('upload_file'):
            return self.upload_file()

    def clear_session(self):
        print("SESSION CLEARED")
        session.clear()
        clear_directory(Config.UPLOAD_FOLDER)
        return '', 200

    def capture_text(self):
        try:
            data = request.json
            page_number = data.get("page_number", 0)
            text = extract_text_from_page(get_session_data("pdf_path"), page_number)
            set_session_data("source_page", page_number)
            set_session_data("source_text", text)
            update_session_messages()
            return jsonify({"message": "{}".format(str(text))})
        except Exception as e:
            print(f"Capture Text Error: {e}")
            return jsonify({"error": str(e)}), 500

    def upload_file(self):
        file = request.files['file']
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            set_session_data("pdf_path", file_path)
            file.save(file_path)
            set_session_data("source_text", extract_first_page_text(get_session_data("pdf_path")))
            update_session_messages()
            file_url = url_for('uploaded_file', filename=filename, _external=True)
            return jsonify({"status": "success", "url": file_url})

        return jsonify({"status": "error", "message": "Invalid file"}), 400

    @classmethod
    def uploaded_file(cls, filename):
        return send_from_directory(Config.UPLOAD_FOLDER, filename)

    def get_uploaded_file(self, filename):
        return self.uploaded_file(filename)

    def dispatch_request(self, *args, **kwargs):
        if request.method == 'GET' and request.path.startswith('/uploads/'):
            return self.get_uploaded_file(*args, **kwargs)
        return super().dispatch_request(*args, **kwargs)

def initialize_routes(app):
    memory_view = MemoryView.as_view('memory_view')
    app.add_url_rule('/', view_func=ChatView.as_view('index'))
    app.add_url_rule('/capture_text', methods=['POST'], view_func=memory_view)
    app.add_url_rule('/clear_session', methods=['POST'], view_func=memory_view)
    app.add_url_rule('/upload_file', methods=['POST'], view_func=memory_view)
    app.add_url_rule('/uploads/<filename>', methods=['GET'], view_func=memory_view)

