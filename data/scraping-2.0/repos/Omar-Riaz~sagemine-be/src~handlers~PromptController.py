from flask import Blueprint, jsonify, request, Response
from flask.views import MethodView
from openai.error import OpenAIError
from services.prompts.PromptService import PromptService
from models.Sentence import Sentence

prompt_controller = Blueprint('prompt_controller', __name__)

class PromptController(MethodView):

    def __init__(self):
        self.prompt_service = PromptService()

# @prompt_controller.route('/prompt', methods=['POST'])
    def post(self, student_id: str) -> Response:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON or wrong content-type.'}), 400
        if 'prompt' not in data:
            return jsonify({'error': 'No prompt provided.'}), 400

        prompt = data['prompt']
        try:
            if 'create a diagram of the process for' in prompt.lower():
                prompt = prompt.lower()
                prompt = prompt.replace('create a diagram of the process for', '')
                answer = self.prompt_service.answer_prompt(prompt, student_id, "services/prompts/MermaidTemplate.txt", False)
                answer.append(Sentence(source=answer[-1].source, string="A diagram has been generated, please refer to the diagram generator below", isDiagram=True))
            else:
                answer = self.prompt_service.answer_prompt(prompt, student_id, "services/prompts/PromptWithEmbeddingTemplate.txt", True)
            return jsonify([sentence.dict() for sentence in answer]), 200
        except OpenAIError as err:
            return jsonify({'error': str(err)}), 500


prompt_controller.add_url_rule('/prompt/<student_id>', view_func=PromptController.as_view('answer_prompt'), methods=['POST'])