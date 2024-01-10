import grpc
import autoReplyService_pb2
import autoReplyService_pb2_grpc

from concurrent import futures
from answerGenerator import generateAnswer
import openai


class QuestionReplyServicer(autoReplyService_pb2_grpc.QuestionReplyer):
    def replyQuestion(self, request, context):
        openai.api_key = request.userApiKey
        question = request.questionMessage
        answer = generateAnswer(question)
        return autoReplyService_pb2.Reply(replyMessage=answer)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    autoReplyService_pb2_grpc.add_QuestionReplyerServicer_to_server(
        QuestionReplyServicer(), server
    )
    server.add_insecure_port("localhost:7337")
    print("Server Started http://localhost:7337")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
