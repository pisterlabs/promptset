from langchain.prompts.chat import HumanMessagePromptTemplate


class Observation(Exception):
    def __init__(self, *, description=None, step_nr=None, add_step_nr=None, target_phase=None,
                 plan_step_info=None):
        self.description = description
        self.step_nr = step_nr
        self.add_step_nr = add_step_nr
        self.target_phase = target_phase
        self.handled = False
        self.plan_step_info = plan_step_info

    def set_target_phase(self, target_phase):
        self.target_phase = target_phase

    def __str__(self):
        result = f"Observation{(' from Step ' + str(self.step_nr)) if self.step_nr is not None else ''}. "
        if self.description:
            result += self.description
        return result

    def set_step_number(self, i):
        if self.add_step_nr:
            self.step_nr = i

    def get_message(self, suffix=None):
        msg = str(self)
        if suffix:
            msg += " " + suffix
        prompt = HumanMessagePromptTemplate.from_template(msg)
        if len(prompt.input_variables) == 0:
            return prompt.format()
        return prompt


class ExecutionError(Observation):

    def __init__(self, *, description=None, step_nr=None, original_error=None, add_step_nr=True, target_phase=None):
        self.description = description
        self.original_error = original_error
        self.step_nr = step_nr
        self.add_step_nr = add_step_nr
        self.fix_idea = None
        self.target_phase = target_phase
        self.handled = False

    def set_fix_idea(self, fix_idea):
        self.fix_idea = fix_idea

    def __str__(self):
        result = f"Something went wrong{(' in Step ' + str(self.step_nr)) if self.step_nr is not None else ''}! "
        if self.original_error:
            result += f"{type(self.original_error).__name__}({self.original_error}). "
        if self.description:
            result += self.description + " "
        if self.fix_idea:
            result += "\nThis is how it can be fixed: " + self.fix_idea
        return result.strip()

class PlanFinished(Observation):
    def __init__(self):
        pass

    def __str__(self):
        return "Plan successfully executed!"
