"""Controller

This file should be imported as a module and contains the followings functions:

    * buildRequest - builds the request from a request object
"""
from enum import Enum
from openaiclient.model.request.completion import CompletionRequest
from openaiclient.model.request.edit import EditRequest
from openaiclient.model.models import Models
from openaiclient.model.response import Response
from openaiclient.view.view import View


class Controller:
    """
    This class manages the data model and the view model.

    ...

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, api):
        # initialized variables

        # different models we can use
        self._models = Models()
        # the view
        self._view = View(self)

        self._module = api

        # uninitialized variables

        # the request we are making
        self._request = None
        # the response
        self._response = None

    def start(self, startVal) -> None:
        match startVal:
            case StartKey.GOOD_START:
                self.view.mainWindow()
            case StartKey.NO_API_KEY:
                self.view.apiWindow()

    @property
    def models(self):
        return self._models

    @property
    def view(self):
        return self._view

    @property
    def request(self):
        return self._request

    def compReq(self) -> None:
        self._request = CompletionRequest(self._module, self.models)

    def editReq(self) -> None:
        self._request = EditRequest(self._module, self.models)

    @property
    def response(self):
        return self._response

    # TODO: implement functions for this class.
    """
    # creates a request
    def buildRequest(self, req):
       Creates a request
        self.request = self.view.getRequest(req)
        self.response = self.request.getResponse()

        self.view.updateResponse(self.response)

    # build an edit request
    def buildEditRequest(self):
        if not isinstance(self.request, EditRequest):
            self.request = EditRequest()
        self.buildRequest(self.request)

    # build a completion request
    def buildCompletionRequest(self):
        if not isinstance(self.request, CompletionRequest):
            self.request = CompletionRequest()
        self.buildRequest(self.request)

    # reset the current request
    def resetRequest(self):
        if isinstance(self.request, CompletionRequest):
            self.request = CompletionRequest()
        else:
            self.request = EditRequest()

    ""Request Settings Data""

    def getSettings(self, className):
        if not isinstance(self.request, className):
            # return an error
            pass

        return self.request.getSettings()

    # returns the keys that are settings for
    # the completion request
    def getCompletionSettings(self):
        return self.getSettings(CompletionRequest)

    # returns the keys that are settings for
    # the completion request
    def getEditSettings(self):
        return self.getSettings(EditRequest)
    """

class StartKey(Enum):
    GOOD_START = 0
    NO_API_KEY = 1


if __name__ == "__main__":

    controller = Controller()

    nextView = controller.initView()

    while True:
        nextView = controller.getView(nextView)
