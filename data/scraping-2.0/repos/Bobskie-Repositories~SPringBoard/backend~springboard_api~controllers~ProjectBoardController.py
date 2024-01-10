import json
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework import generics
from rest_framework.response import Response
from rest_framework import status
from springboard_api.serializers import ProjectBoardSerializer
from springboard_api.models import ProjectBoard, Project
import requests
from django.db.models import Max
from django.conf import settings
import os
from openai import OpenAI


class CreateProjectBoard(generics.CreateAPIView):
    serializer_class = ProjectBoardSerializer

    def perform_create(self, serializer, data):
        serializer.save(**data)

    def update_project_score(self, project, add_score):
        project.score += add_score
        project.save()

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        data = {}

        highest_board_id = ProjectBoard.objects.aggregate(Max('boardId'))[
            'boardId__max']
        new_board_id = highest_board_id + 1 if highest_board_id is not None else 1

        # api_url = "https://api.openai.com/v1/engines/text-davinci-003/completions"
        prompt = (
            f"Please analyze the following data: {request.data.get('content', '')}. "
            f"Provide a detailed and critical rating (1-10) in numerical value(not in string) for the following aspects: "
            f"\n1. Novelty: Evaluate the originality of the data. "
            f"\n2. Technical Feasibility: Assess whether the data is technically sound and feasible. "
            f"\n3. Capability: Determine if the data demonstrates capability. "
            f"\nRatings below 5 should be considered for data that lacks composition, effort, verbosity, or information. "
            f"Be critical and practical when rating. "
            f"Include at least 2 specific sentences of advice for improvements (Recommendations) and "
            f"2 sentences of feedback on how the data is presented and structured, and what can be done to improve those aspects (Feedback) for each of the above aspects. "
            f"The output should be in the following JSON format: "
            f"\n'novelty': 'numerical rating', 'technical_feasibility': 'numerical rating', 'capability': 'numerical rating', "
            f"'recommendations_novelty': ['specific advice'], 'recommendations_technical_feasibility': [' advice'], "
            f"'recommendations_capability': ['specific advice'], 'feedback_novelty': ['specific feedback'], "
            f"'feedback_technical_feasibility': ['feedback'], 'feedback_capability': ['specific feedback']. "
            f"Ensure a fair and balanced assessment for each aspect."
        )

        # request_payload = {
        #     "prompt": prompt,
        #     "temperature": 0.5,
        #     "max_tokens": 256,
        #     "top_p": 1.0,
        #     "frequency_penalty": 0.0,
        #     "presence_penalty": 0.0
        # }

        # headers = {"Authorization": os.environ.get("OPENAI_KEY", "")}
        client = OpenAI(api_key=os.environ.get("OPENAI_KEY", ""))
        message = [
            {"role": "user", "content": prompt}
        ]

        try:
            # response = requests.post(
            # api_url, json=request_payload, headers=headers)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", messages=message, temperature=0, max_tokens=1050
            )
            if response:
                try:
                    # response_content = response.json()
                    # print(response_content)
                    choices = response.choices
                    first_choice_content = response.choices[0].message.content
                    # print(first_choice_content)

                    if choices:
                        # gpt_response = choices[0]["text"].strip()
                        gpt_response = first_choice_content
                        json_response = json.loads(gpt_response)
                        print(json_response)
                        novelty = json_response.get("novelty", 0)
                        technical_feasibility = json_response.get(
                            "technical_feasibility", 0)
                        capability = json_response.get("capability", 0)

                        # recommendations = ' '.join(
                        # json_response.get("recommendations", []))
                        # feedback = ' '.join(json_response.get("feedback", []))

                        recommendations_novelty = json_response.get(
                            "recommendations_novelty", [])
                        recommendations_technical_feasibility = json_response.get(
                            "recommendations_technical_feasibility", [])
                        recommendations_capability = json_response.get(
                            "recommendations_capability", [])

                        feedback_novelty = json_response.get(
                            "feedback_novelty", [])
                        feedback_technical_feasibility = json_response.get(
                            "feedback_technical_feasibility", [])
                        feedback_capability = json_response.get(
                            "feedback_capability", [])

                        recommendations = '\n'.join([
                            "Novelty Recommendations:\n" +
                            '\n'.join(recommendations_novelty),
                            "\n\nTechnical Feasibility Recommendations:\n" +
                            '\n'.join(
                                recommendations_technical_feasibility),
                            "\n\nCapability Recommendations:\n" +
                            '\n'.join(recommendations_capability)
                        ])

                        feedback = '\n'.join([
                            "Novelty Feedback:\n" +
                            '\n'.join(feedback_novelty),
                            "\n\nTechnical Feasibility Feedback:\n" +
                            '\n'.join(feedback_technical_feasibility),
                            "\n\nCapability Feedback:\n" +
                            '\n'.join(feedback_capability)
                        ])

                        # reference_links = ', '.join(
                        #   json_response.get("references", []))

                        # if not (reference_links.startswith('"') and reference_links.endswith('"')):
                        #     reference_links = f'{reference_links}'

                        title = request.data.get('title', '')
                        content = request.data.get('content', '')
                        project_fk_id = request.data.get('project_fk', None)

                        data = {
                            'title': title,
                            'content': content,
                            'novelty': novelty,
                            'technical_feasibility': technical_feasibility,
                            'capability': capability,
                            'recommendation': recommendations,
                            'feedback': feedback,
                            # 'references': reference_links,
                            'project_fk': Project.objects.get(id=project_fk_id),
                            'boardId': new_board_id,
                        }

                        project_instance = Project.objects.get(
                            id=project_fk_id)
                        add_score = (
                            (novelty * 0.4) +
                            (technical_feasibility * 0.3) +
                            (capability * 0.3)
                        )
                        self.update_project_score(
                            project_instance, add_score)

                    else:
                        print("No response content or choices found.")
                except json.JSONDecodeError as json_error:
                    return Response({"error": f"Error decoding JSON response: {json_error}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                return Response({"error": response.text}, status=status.HTTP_400_BAD_REQUEST)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            data = {}

        if serializer.is_valid():
            self.perform_create(serializer, data)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class GetProjectBoards(generics.ListAPIView):
    serializer_class = ProjectBoardSerializer

    def get_queryset(self):
        project_id = self.kwargs.get('project_id')

        # Get the latest distinct project boards for each templateId within the specified project
        queryset = ProjectBoard.objects.filter(project_fk_id=project_id).values(
            'templateId').annotate(
                latest_id=Max('id'),
        ).values(
                'latest_id',
        )

        return ProjectBoard.objects.filter(id__in=queryset)

    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class GetVersionProjectBoards(generics.ListAPIView):
    serializer_class = ProjectBoardSerializer
    queryset = ProjectBoard.objects.all()

    def get(self, request, *args, **kwargs):
        projectboard_id = self.kwargs.get('projectboard_id')

        try:
            projectboard = ProjectBoard.objects.get(id=projectboard_id)
            template_id = projectboard.templateId
            board_id = projectboard.boardId

            # Retrieve related project boards with the same templateId and boardId
            related_projectboards = ProjectBoard.objects.filter(
                templateId=template_id, boardId=board_id)

            # Sort the related project boards in decreasing order of their creation date
            related_projectboards = related_projectboards.order_by(
                '-created_at')

            serializer = ProjectBoardSerializer(
                related_projectboards, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except ProjectBoard.DoesNotExist:
            return Response({"error": "ProjectBoard not found"}, status=status.HTTP_404_NOT_FOUND)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class GetProjectBoardById(generics.ListAPIView):
    serializer_class = ProjectBoardSerializer
    queryset = ProjectBoard.objects.all()

    def get(self, request, *args, **kwargs):
        projectboard_id = self.kwargs.get('projectboard_id')

        try:
            projectboard = ProjectBoard.objects.get(id=projectboard_id)
            serializer = ProjectBoardSerializer(projectboard)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except ProjectBoard.DoesNotExist:
            return Response({"error": "ProjectBoards not found"}, status=status.HTTP_404_NOT_FOUND)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class UpdateBoard(generics.CreateAPIView):
    serializer_class = ProjectBoardSerializer

    def update_project_score(self, project, subtract_score, new_score):
        project.score -= subtract_score
        project.score += new_score
        project.save()

    def create(self, request, *args, **kwargs):
        data = request.data
        project_board_id = kwargs.get('projectboard_id')

        try:
            project_board = ProjectBoard.objects.get(id=project_board_id)

            subtract_score = (
                (project_board.novelty * 0.4) +
                (project_board.technical_feasibility * 0.3) +
                (project_board.capability * 0.3)
            )

            # api_url = "https://api.openai.com/v1/engines/text-davinci-003/completions"
            prompt = (
                f"Please analyze the following data: {request.data.get('content', '')}. "
                f"Provide a detailed and critical rating (1-10) in numerical value(not in string) for the following aspects: "
                f"\n1. Novelty: Evaluate the originality of the data. "
                f"\n2. Technical Feasibility: Assess whether the data is technically sound and feasible. "
                f"\n3. Capability: Determine if the data demonstrates capability. "
                f"\nRatings below 5 should be considered for data that lacks composition, effort, verbosity, or information. "
                f"Be critical and practical when rating. "
                f"Include at least 2 specific sentences of advice for improvements (Recommendations) and "
                f"2 sentences of feedback on how the data is presented and structured, and what can be done to improve those aspects (Feedback) for each of the above aspects. "
                f"The output should be in the following JSON format: "
                f"\n'novelty': 'numerical rating', 'technical_feasibility': 'numerical rating', 'capability': 'numerical rating', "
                f"'recommendations_novelty': ['specific advice'], 'recommendations_technical_feasibility': ['advice'], "
                f"'recommendations_capability': ['specific advice'], 'feedback_novelty': ['specific feedback'], "
                f"'feedback_technical_feasibility': ['feedback'], 'feedback_capability': ['specific feedback']. "
                f"Ensure a fair and balanced assessment for each aspect."
            )

            # request_payload = {
            #     "prompt": prompt,
            #     "temperature": 0.5,
            #     "max_tokens": 256,
            #     "top_p": 1.0,
            #     "frequency_penalty": 0.0,
            #     "presence_penalty": 0.0
            # }

            # headers = {
            #     "Authorization": os.environ.get("OPENAI_KEY") + ""
            # }

            # response = requests.post(
            #     api_url, json=request_payload, headers=headers)
            client = OpenAI(api_key=os.environ.get("OPENAI_KEY", ""))
            message = [
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", messages=message, temperature=0, max_tokens=1050
            )
            if response:
                try:
                    # response_content = response.json()
                    # choices = response_content.get("choices", [])
                    choices = response.choices
                    first_choice_content = response.choices[0].message.content
                    if choices:
                        # gpt_response = choices[0]["text"].strip()
                        gpt_response = first_choice_content
                        json_response = json.loads(gpt_response)
                        # print(json_response)
                        novelty = json_response.get("novelty", 0)
                        technical_feasibility = json_response.get(
                            "technical_feasibility", 0)
                        capability = json_response.get("capability", 0)
                        recommendations_novelty = json_response.get(
                            "recommendations_novelty", [])
                        recommendations_technical_feasibility = json_response.get(
                            "recommendations_technical_feasibility", [])
                        recommendations_capability = json_response.get(
                            "recommendations_capability", [])

                        feedback_novelty = json_response.get(
                            "feedback_novelty", [])
                        feedback_technical_feasibility = json_response.get(
                            "feedback_technical_feasibility", [])
                        feedback_capability = json_response.get(
                            "feedback_capability", [])

                        recommendations = '\n'.join([
                            "Novelty Recommendations:\n" +
                            '\n'.join(recommendations_novelty),
                            "\n\nTechnical Feasibility Recommendations:\n" +
                            '\n'.join(
                                recommendations_technical_feasibility),
                            "\n\nCapability Recommendations:\n" +
                            '\n'.join(recommendations_capability)
                        ])

                        feedback = '\n'.join([
                            "Novelty Feedback:\n" +
                            '\n'.join(feedback_novelty),
                            "\n\nTechnical Feasibility Feedback:\n" +
                            '\n'.join(feedback_technical_feasibility),
                            "\n\nCapability Feedback:\n" +
                            '\n'.join(feedback_capability)
                        ])

                        # recommendations = ' '.join(
                        #  json_response.get("recommendations", []))
                        # feedback = ' '.join(json_response.get("feedback", []))
                        # reference_links = ', '.join(
                        #    json_response.get("references", []))

                        # if not (reference_links.startswith('"') and reference_links.endswith('"')):
                        #     reference_links = f'{reference_links}'

                        data = {
                            'title': data.get('title', ''),
                            'content': data.get('content', ''),
                            'novelty': novelty,
                            'technical_feasibility': technical_feasibility,
                            'capability': capability,
                            'recommendation': recommendations,
                            'feedback': feedback,
                            # 'references': reference_links,
                            'project_fk': project_board.project_fk,
                            'templateId': project_board.templateId,
                            'boardId': project_board.boardId,
                        }

                        new_board_instance = ProjectBoard(**data)
                        new_board_instance.save()

                        project_instance = Project.objects.get(
                            id=project_board.project_fk.id)

                        new_score = (
                            (novelty * 0.4) +
                            (technical_feasibility * 0.3) + (capability * 0.3)
                        )
                        subtract_score = subtract_score

                        self.update_project_score(
                            project_instance, subtract_score, new_score)

                        # if response.status_code != 200:
                        #     return Response({"error": "Failed to update project score"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                    else:
                        return Response({"error": "No response content or choices found"}, status=status.HTTP_400_BAD_REQUEST)
                except json.JSONDecodeError as json_error:
                    return Response({"error": f"Error decoding JSON response: {json_error}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                return Response({"error": response.text}, status=status.HTTP_400_BAD_REQUEST)

        except ProjectBoard.DoesNotExist:
            return Response({"error": "ProjectBoard not found"}, status=status.HTTP_404_NOT_FOUND)
        except requests.exceptions.RequestException as e:
            return Response({"error": f"An error occurred: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"id": new_board_instance.id}, status=status.HTTP_201_CREATED)


class DeleteProjectBoard(generics.DestroyAPIView):
    queryset = ProjectBoard.objects.all()
    serializer_class = ProjectBoardSerializer
    lookup_field = 'id'

    def destroy(self, request, *args, **kwargs):
        try:
            # Use get_object_or_404 for cleaner code
            instance = self.get_object()

            # Calculate subtract_score for the specified project board
            subtract_score = (
                (instance.novelty * 0.4) +
                (instance.technical_feasibility * 0.3) +
                (instance.capability * 0.3)
            )

            # Update the project's score directly in the code
            instance.project_fk.score -= subtract_score
            instance.project_fk.save()

            # Delete all related project boards with the same boardId in a single query
            ProjectBoard.objects.filter(boardId=instance.boardId).delete()

            return Response(status=status.HTTP_204_NO_CONTENT)
        except ProjectBoard.DoesNotExist:
            return Response({"error": "ProjectBoard not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
