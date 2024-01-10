from . import Course
from . import prompt
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class Student:
    number_of_students = 0 # ignore for now

    def __init__(self, name: str, age: int, gender: chr, education_level: str, special_need: str) -> None:
        # Personal Information & Education needs
        self.name = name
        self.age = age
        self.gender = gender
        self.education_level = education_level
        self.special_education_need = special_need
        self.convertion_history = []
        
        # Course
        self.courses_database = dict() # dict(course_name: course object)
        self.current_course_name = ""

    def retrieve_courses_list(self) -> list:
        """return a list that contain all courses that created by the student"""
        return self.courses_database.keys()

    def get_topic_list_of_current_couese(self) -> list:
        """Return the topic of the course that is taking by the student"""
        return self.courses_database[self.current_course_name].get_course_topic()
    
    def delete_course(self, course_name: str = None) -> None:
        """delete the course from the student course database"""
        del self.courses_database['course_name']
    
    def create_course(self, subject: str):
        """Create a new course"""
        new_course = Course.Course(self.education_level, self.special_education_need, subject)
        course_name = new_course.course_name
        self.courses_database[course_name] = new_course
        self.current_course_name = course_name
    
    def course_select(self, user_decision) -> str:
        """Select course to study from the created course"""
        if user_decision in self.courses_database:
            self.current_course_name = user_decision
            return "Successful"
        return "Course does not exist"

    def course_change_current_topic(self, topic_name) -> None:
        """Change the topic to study in the course"""
        return self.courses_database[self.current_course_name].change_topic(topic_name) == True

    def course_speak_with_virtual_teacher(self, user_input) -> str:
        """Communicate with the virtual teacher"""
        LLM = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
        if self.current_course_name == "":
            return "Select what course or create a course first!"
        
        if self.current_course_name not in self.retrieve_courses_list():
            return "The course is not a created course"
        
        course = self.courses_database[self.current_course_name]
        course.weekly_teaching_schedule[f"week_{course.current_week}"]["chat history"].append(HumanMessage(content=user_input))
        response = LLM(course.weekly_teaching_schedule[f"week_{course.current_week}"]["chat history"]).content
        course.weekly_teaching_schedule[f"week_{course.current_week}"]["chat history"].append(AIMessage(content=response))
        return response

    def lesson_custiomized_teaching(self, teacher_speech: str):
        """Take in teacher speech and convert to easier understanding wording"""
        LLM = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
        if len(self.convertion_history) == 0:
            self.convertion_history = prompt.get_teaching_instruction(self.special_education_need, self.education_level)
        
        self.convertion_history.append(HumanMessage(content = teacher_speech))
        response = LLM(self.convertion_history).content
        self.convertion_history.append(AIMessage(content = response))
        return response

