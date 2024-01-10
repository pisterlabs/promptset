# <copyright_statement>
#   CodeBuddy: A programming assignment management system for short-form exercises
#   Copyright (C) 2023 Stephen Piccolo
#   This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details. You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
# </copyright_statement>

from BaseUserHandler import *
import openai

class AskVirtualAssistantHandler(BaseUserHandler):
    async def post(self, course_id, assignment_id, exercise_id):
        out_dict = {"message": "", "success": False}

        try:
            course_details = await self.get_course_details(course_id)

            virtual_assistant_max_per_exercise = course_details["virtual_assistant_config"]["max_per_exercise"]

            if virtual_assistant_max_per_exercise > 0:
                virtual_assistant_interactions = self.content.get_virtual_assistant_interactions(course_id, assignment_id, exercise_id, self.user_info["user_id"])

                if len(virtual_assistant_interactions) > virtual_assistant_max_per_exercise:
                    out_dict["message"] = f"You have reach the limit ({virtual_assistant_max_per_exercise}) for questions that can be asked on this exercise."

            if out_dict["message"] == "":
                question = self.get_body_argument("question").strip()
                student_code = self.get_body_argument("student_code").strip()

                response = await self.access_openai(course_details, course_id, assignment_id, exercise_id, question, student_code)

                self.content.save_virtual_assistant_interaction(course_id, assignment_id, exercise_id, self.get_current_user(), question, response)

                out_dict["message"] = response
                out_dict["success"] = True
        except Exception as inst:
            out_dict["message"] = traceback.format_exc()

        self.write(json.dumps(out_dict, default=str))

    async def access_openai(self, course_details, course_id, assignment_id, exercise_id, question_content, student_code):
        course_basics = await self.get_course_basics(course_id)
        course_id = course_basics["id"]
        assignment_basics = await self.get_assignment_basics(course_basics, assignment_id)
        exercise_details = await self.get_exercise_details(course_basics, assignment_basics, exercise_id)

        openai.api_key = course_details["virtual_assistant_config"]["api_key"]

        # test_prompt = ""
        # for test in exercise_details["tests"]:
        #     print(test)
        #TODO? exercise_details["tests"] (including expected output)
        #TODO? exercise_details["data_files"] (truncate when necessary)

        system_content = f"You are a helpful assistant who gives a hint or short explanation to a student who is writing {exercise_details['back_end'].split('_')[0]} code. Do *not* provide any code. Only provide one or a few sentences to help the student meet the prompt requirements.\n\n"
        
        if len(student_code) > 0:
             system_content += f"Below is the student's current code:\n\n```{student_code}```\n\n"
             
        system_content += f"Below is the prompt that the student is working to address:\n\n{exercise_details['instructions']}"

        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": question_content}]

        response = await openai.ChatCompletion.acreate(
            model="gpt-4-0314",
            temperature=0.7,
            messages=messages,
            timeout=30
        )

        content = response.choices[0].message.content

        # Remove any code fragments
        content = re.sub(r"```[\s\S]*?```", "```redacted```", content)

        return content