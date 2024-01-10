import json
import os

from flask import jsonify, request, Blueprint
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

with open('../config.json') as f:
    config = json.load(f)

os.environ["SERPER_API_KEY"] = config['SERPER_API_KEY']

# from langchain.utilities import GoogleSerperAPIWrapper

bio_blueprint = Blueprint('bio_blueprint', __name__)
# search = GoogleSerperAPIWrapper()
chat = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=config['openai_api-key'])


query = """
<!-- Modal -->

<!-- Modal -->
    <div class="modal fade" id="viewContactModal" tabindex="-1" aria-labelledby="viewContactModalLabel" aria-hidden="true">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="viewContactModalLabel">Account History</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <!-- Tabs -->
                    <ul class="nav nav-tabs" id="myTab" role="tablist">
                        <li class="nav-item" role="presentation">
                            <a class="nav-link active" id="calls-tab" data-bs-toggle="tab" href="#calls" role="tab" aria-controls="calls" aria-selected="true">Calls</a>
                        </li>
                        <li class="nav-item" role="presentation">
                            <a class="nav-link" id="video-meetings-tab" data-bs-toggle="tab" href="#video-meetings" role="tab" aria-controls="video-meetings" aria-selected="false">Video Meetings</a>
                        </li>
                        <li class="nav-item" role="presentation">
                            <a class="nav-link" id="contact-recommendations-tab" data-bs-toggle="tab" href="#contact-recommendations" role="tab" aria-controls="contact-recommendations" aria-selected="false">Contact Recommendations</a>
                        </li>
                    </ul>
                    <div class="tab-content" id="myTabContent">
                        <div class="tab-pane fade show active" id="calls" role="tabpanel" aria-labelledby="calls-tab">
                            <!-- Calls content goes here -->
                            <p>No calls to display</p>
                        </div>
                        <div class="tab-pane fade" id="video-meetings" role="tabpanel" aria-labelledby="video-meetings-tab">
                            <!-- Video meetings content goes here -->
                            <p>No video meetings to display</p>
                        </div>
                        <div class="tab-pane fade" id="contact-recommendations" role="tabpanel" aria-labelledby="contact-recommendations-tab">
                            <!-- Contact recommendations content goes here -->
                            <p>No contact recommendations to display</p>
                        </div>
                    </div>
                </div>
                <div class="modal-footer d-flex justify-content-center" style="border-top: 1px solid gray;">
                    <button type="button" class="btn btn-secondary mx-2"><i class="fas fa-phone"></i></button>
                    <button type="button" class="btn btn-secondary mx-2"><i class="fas fa-video"></i></button>
                    <button type="button" class="btn btn-secondary mx-2"><i class="fas fa-envelope"></i></button>
                </div>
            </div>
        </div>
    </div>

Thank you for this code above. It worked perfectly. One addition I need you to make is make the icon images appear. they are just gray rectangles right now. also center and space out the icons.

Let's implement these changes:

"""

# results = search.results(query)
template = (
    "You are a helpful assistant that takes in a user query and responds to it in a helpful way. "
    "User Query: {text}."
)
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

# get a chat completion from the formatted messages
answer = chat(
    chat_prompt.format_prompt(
        text=f"{query}"
    ).to_messages()
)

print(answer.content)
# return answer.content