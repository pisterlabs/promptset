# test_completionrequest.py
import unittest
from tests.unit.fixture import api
from openaiclient.model.request.edit import EditRequest
from openaiclient.model.response import Response
"""
Class TestCompletionRequest:

Tests different units of the CompletionRequest class.
"""


class TestEditRequest(unittest.TestCase):
    def setUp(self):
        self.request = EditRequest(api)
        self.REQUIRED_ARGS = 2

    # test that the requestDict var is not empty
    # (should be setup)
    def test_RequestNotEmpty(self):
        self.assertTrue(self.request.requestDict)

    # test that the requiredArgs var is not empty
    # (should be setup)
    def test_RequiredArgsNotEmpty(self):
        self.assertTrue(self.request.requiredArgs)

    # test that the optionalArgs var is not empty
    # (should be setup)
    def test_OptionalArgsNotEmpty(self):
        self.assertTrue(self.request.optionalArgs)

    # test that the settings var is not empty
    # (should be setup)
    def test_SettingsNotEmpty(self):
        self.assertTrue(self.request.settings)

    # if the following three tests pass then
    # the optionalArgs variable is also correct.

    # test that the requiredArgs variable and the
    # optionalArgs variable are disjoint
    def test_RequiredOptionalDisjoint(self):
        self.assertTrue(
            self.request.requiredArgs.isdisjoint(
                self.request.optionalArgs
            )
        )

    # test that requiredArgs has the correct
    # number of elements
    def test_RequiredArgsCorrectLength(self):
        self.assertIs(len(self.request.requiredArgs),
                      self.REQUIRED_ARGS)

    # test that requiredArgs has the correct
    # values
    def test_RequiredArgsCorrectValue(self):
        self.assertIn("model", self.request.requiredArgs)
        self.assertIn("instruction", self.request.requiredArgs)

    # # #

    # tests that the correct values are in
    # the settings
    def test_SettingsCorrectValue(self):
        self.assertNotIn("instruction", self.request.settings)
        self.assertNotIn("input", self.request.settings)

    # test that the obect returned from getResponse() is
    # a Response object.
    def test_GetResponse(self):
        self.assertIsInstance(self.request.getResponse(), Response)


if __name__ == "__main__":
    unittest.mainloop()
