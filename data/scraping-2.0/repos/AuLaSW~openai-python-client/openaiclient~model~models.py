import openai

"""
Class Models

This class retrieves the models from the OpenAI API
and creates a dictionary of the usable classes.
"""


class Models:
    def __init__(self):
        self.COMPLETION = "Completion"
        self.EDIT = "Edit"

        # dictionary of models
        self._models = {
            "text-davinci-003": Model(
                name="text-davinci-003",
                max_tokens=4_000,
                type=self.COMPLETION
            ),
            "text-curie-001": Model(
                name="text-curie-001",
                max_tokens=2_048,
                type=self.COMPLETION
            ),
            "text-babbage-001": Model(
                name="text-babbage-001",
                max_tokens=2_048,
                type=self.COMPLETION
            ),
            "text-ada-001": Model(
                name="text-ada-001",
                max_tokens=2_048,
                type=self.COMPLETION
            ),
            "text-davinci-edit-001": Model(
                name="text-davinci-edit-001",
                type=self.EDIT
            ),
        }

    def __str__(self):
        strModels = "Usable Models:\n"

        strModels += "  " + self.COMPLETION + " Models:\n"

        for _, model in self.models.items():
            if model.type == self.COMPLETION:
                strModels += "    " + model.name + "\n"

        strModels += "  " + self.EDIT + " Models:\n"

        for _, model in self.models.items():
            if model.type == self.EDIT:
                strModels += "    " + model.name + "\n"

        return strModels

    def __contains__(self, item):
        return item in self._models

    @property
    def models(self):
        return self._models

    @property
    def completionModels(self):
        modelDict = {}
        for key, model in self.models.items():
            if model.type == self.COMPLETION:
                modelDict[key] = model

        return modelDict

    @property
    def editModels(self):
        modelList = []
        for _, model in self.models.items():
            if model.type == self.EDIT:
                modelList.append(model)

        return modelList

    @property
    def text_davinci_003(self):
        return self.models["text-davinci-003"]

    @property
    def text_curie_001(self):
        return self.models["text-curie-001"]

    @property
    def text_babbage_001(self):
        return self.models["text-babbage-001"]

    @property
    def text_ada_001(self):
        return self.models["text-ada-001"]

    @property
    def text_davinci_edit_001(self):
        return self.models["text-davinci-edit-001"]


class Model:
    def __init__(self, name, type, max_tokens=0):
        self._name = name
        self._type = type

        # Edit types don't have a max_tokens attribute,
        # so set to -1 to make sure it can't be used.
        if self._type == "Edit":
            self._max_tokens = -1
        else:
            self._max_tokens = max_tokens

    @property
    def name(self):
        return self._name

    @property
    def max_tokens(self):
        return self._max_tokens

    @property
    def type(self):
        return self._type


if __name__ == "__main__":
    model = Models()

    print(model)
