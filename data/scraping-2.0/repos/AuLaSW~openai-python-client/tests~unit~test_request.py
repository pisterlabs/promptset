# test_request.py
import unittest
from openaiclient.model.request.request import Request


class TestRequest(unittest.TestCase):
    def setUp(self):
        self.request = Request()

    # tests that the requestDict is empty on
    # instantiation
    def test_RequestEmpty(self):
        self.assertFalse(self.request.requestDict)

    # test that requiredArgs is empty
    def test_RequiredArgsEmpty(self):
        self.assertFalse(self.request.requiredArgs)

    # test that optionalArgs is empty
    def test_OptionalArgsEmpty(self):
        self.assertFalse(self.request.optionalArgs)

    # test that settings is empty
    def test_EmptySettingsRaiseError(self):
        with self.assertRaises(RuntimeError) as error:
            self.request.settings

        self.assertIsInstance(error.exception, RuntimeError)

    # test that getResponse() returns a NotImplementedError
    def test_GetResponse(self):
        self.assertRaises(
            NotImplementedError,
            self.request.getResponse)

    # test that getKeys() called with an empty dictionary
    # will return a runtime error.
    def test_GetKeys(self):
        self.assertRaises(RuntimeError, self.request.getKeys)


if __name__ == "__main__":
    unittest.mainloop()
