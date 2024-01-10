from action import Actioner, Utils
from grafics import GraphicsTools
from configuration import Configuration
from openai_conn import OpenAIConn
from print_handler import PrintHandler as handler

class Tasker(OpenAIConn):
    __system_message = Configuration.SYSTEM_MESSAGE_TASKER
    
    def __init__(self, image_name, test_mode=False):
        super().__init__(self.__system_message)
        self.__actioner = Actioner()
        self.__tester = TaskerTest()
        self.__util = Utils()
        self.__print_handler = handler()
        self.__image_name = image_name
        self.__test_mode = test_mode

    def create_normal_tasks(self, super_task):
        try:
            self.__print_handler.print_loading()
            todo = self._create_text_completion(super_task)
            self.__print_handler.break_loading()
            self.__print_handler.print_super_task(super_task, len(todo["list"]))
            self.traverse_and_test(todo["list"], todo["super-task"], todo["super-test"])
        except Exception as e:
            print(e)
            if "No JSON found" in str(e):
                self.create_normal_tasks(super_task)

    def traverse_and_test(self, tasks, super_task, super_test):
        for task in tasks:
            handler().print_task(task["id"], task["task"])
            if task["super-task"]:
                self.create_normal_tasks(task["task"])
            else:
                self.create_primitive_task(task)
                if not self.__test_mode:
                    continue

                next_task_id = task["id"] + 1
                if next_task_id < len(tasks):
                    next_task = tasks[next_task_id]
                    response = self.__tester.visual_confirmation(task["task"], task["test"], next_task["task"], self.__image_name)
                else:
                    response = self.__tester.visual_confirmation(task["task"], task["test"], "No more tasks", self.__image_name)
            
            if not response["task_done"]:
                self.create_vision_tasks(super_task)

            if response["new_task"]:
                self.create_normal_tasks(response["new_task"])
                    
        response = self.__tester.visual_confirmation(super_task, super_test, "No more tasks", self.__image_name)
        if response["new_task"]:
            self.create_normal_tasks(response["new_task"])

    def create_primitive_task(self, task):
        if task["action"] == "Move":
            #self.__actioner.moveUsingAxis(task["x"], task["y"])
            pass
        elif task["action"] == "Press":
            self.__actioner.press_keys(task["key"])
        elif task["action"] == "Type":
            self.__actioner.type_text(task["text"])
        elif task["action"] == "Wait":
            self.__actioner.wait(2)
        elif task["action"] == "Execute":
            self.__actioner.execute(task["command"])

    def create_vision_tasks(self, super_task):
        try:
            self.__print_handler.print_loading()
            self.__actioner.take_screenshot(self.__image_name)
            base64_image = self.__util.encode_image(self.__image_name)
            todo = self._create_visual_completion(super_task, base64_image)
            print(todo)
            self.__print_handler.break_loading()
            self.__print_handler.print_super_task(super_task, len(todo["list"]))
            self.traverse_and_test(todo["list"], todo["super-task"], todo["super-test"])
        except Exception as e:
            print(e)
            if "No JSON found" in str(e):
                self.create_vision_tasks(super_task + " (Avoid create a large list, use super-taks)")
            
class VisualAI(OpenAIConn):
    __system_message = Configuration.SYSTEM_MESSAGE_VISUAL

    def __init__(self):
        super().__init__(self.__system_message)
        self.__actioner = Actioner()
        self.__util = Utils()

    def locate_coordinates(self, super_task, image_name):
        graphic_tools = GraphicsTools(image_name)
        graphic_tools.add_grid()
        base64_image = self.__util.encode_image(image_name)
        todo = self._create_visual_completion(super_task, base64_image)
        print(todo)
        (x, y) = (todo["coordinates"]["x"], todo["coordinates"]["y"])
        graphic_tools.insert_mouse_cursor(x, y)

class TaskerTest(OpenAIConn):
    __system_message = Configuration.SYSTEM_MESSAGE_TESTER

    def __init__(self):
        super().__init__(self.__system_message)
        self.__util = Utils()

    def visual_confirmation(self, super_task, super_test, next_task, image_name):
        base64_image = self.__util.encode_image(image_name)
        prompt = f"prompt-super-task: {super_task}\n super-test: {super_test}\n next-task: {next_task}"
        response = self._create_visual_completion(prompt, base64_image)
        print(response)
        return response
