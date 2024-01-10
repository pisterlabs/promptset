from flask import Blueprint, request
from utils.ResponseUtil import ResponseUtil
from service.ChatBotService import ChatBotService
from bson import ObjectId
import openai

chatbot = Blueprint("chatbot", __name__)
userLifestyleService = ChatBotService()

class ChatBotEndpoint:

    @staticmethod
    @chatbot.route("/chatbot", methods=["POST"])
    def getChatbotResponse():
        messagesHistory = request.json["message"]
        chatbotService = ChatBotService()
        message, dietPlan, trainingPlan, maintananceCalories, goal, cutBulkRate, workoutExperience, healthIssues  = chatbotService.interactWithChatbot(messagesHistory)
        return ResponseUtil.buildResponse({
            "role": "assistant", 
            "content": message, 
            "dietPlan": dietPlan, 
            "trainingPlan": trainingPlan, 
            "maintananceCalories": maintananceCalories,
            "goal": goal,
            "cutBulkRate": cutBulkRate,
            "workoutExperience": workoutExperience,
            "healthIssues": healthIssues
        })
