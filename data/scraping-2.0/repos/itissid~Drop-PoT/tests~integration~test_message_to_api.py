import os
import unittest
from typing import Any, Dict, List, Optional, Tuple

import openai
from dotenv import load_dotenv

from drop_backend.lib.ai import AIDriver, AltAI, driver_wrapper
from drop_backend.lib.event_node_manager import BaseEventManager, EventManager
from drop_backend.model.ai_conv_types import (
    MessageNode,
    Role,
    UserExplicitFunctionCall,
)

# Note: If we don't import the full path then the isinstance(weather_obj,
# WeatherEvent) returns False in the __eq__ depending on how the weather_obj was
# created.
from tests.integration.fixtures.weather_event import WeatherEvent

from .fixtures.schema.weather_event_schema import (
    weather_event_function_call_param,
)


class NoFunctionCallEventManager(BaseEventManager):
    def get_function_call_spec(
        self,
    ):
        return None, None

    def extract_fn_name(self, ai_message: MessageNode) -> Optional[str]:
        return None

    def extract_fn_args(
        self, ai_message: MessageNode
    ) -> Tuple[List[Any], Dict[str, Any]]:
        return [], {}

    def should_call_function(self, ai_message: MessageNode) -> bool:
        return False

    def call_fn_by_name(self, fn_name: str, *args, **kwargs):
        return None, None


class TestSendToOpenAIAPI(unittest.TestCase):
    def test_replay_messages_for_function_call_are_separately_maintained(self):
        """
        - Send a user message: "Whats the weather in Boston in farenheit?" with
        a function call to the AI using driver_wrapper.
        - Along with the function call to the AI there should be a message with
        the role `function` and the call result.
        - The next message is a correction to the earlier message like "Could
        you specify the weather in Boston in celsius?"
        """
        # Mock out the call to AI and extract the messages and make sure the
        pass

    def test_no_function_execution(self) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = api_key
        # Lets make a test spec

        event_manager = NoFunctionCallEventManager()
        driver = driver_wrapper(
            events=[
                "What's the climate typically like in Boston during October"
            ],
            system_message=MessageNode(
                role=Role.system,
                message_content="You are helpful assistant. Follow the instructions I give you. Do not respond until I ask you a question.",
            ),
            ai_driver=AIDriver(AltAI(model="gpt-3.5-turbo-16k"), event_manager),
            event_manager=event_manager,
            user_message_prompt_fn=lambda x: x.raw_event_str,
        )
        event, _ = next(driver)
        assert event.history
        self.assertEqual(len(event.history), 3)
        self.assertEqual(event.history[0].role, Role.system)
        self.assertEqual(event.history[1].role, Role.user)
        self.assertEqual(event.history[2].role, Role.assistant)
        assert event.history[2].message_content
        self.assertGreaterEqual(len(event.history[2].message_content), 1)

        print(event.history[2].message_content)
        # No function call
        self.assertEqual(event.history[2].ai_function_call, None)

    def test_event_to_open_ai__user_function_mandate_is_obeyed(self) -> None:
        """If MessageNode.role == user and MessageNode.message_function_call and MessageNode.explicit_function_call_spec are not null then the
        next message must have a role `function` with call result. The message after that is  an assistant message.

        This is done by calling the AI and getting the actual response from it.

        As an example lets test the weather function.
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = api_key
        import sys

        print(sys.path)
        print(", ".join([i for i in sys.modules.keys() if "drop_backend" in i]))
        event_manager = EventManager(
            "WeatherEvent",
            "tests.integration.fixtures",
            "tests.integration.fixtures.schema",
        )

        # def weather_fn_call_wrapper(ai_message: MessageNode) -> Tuple[Any, str]:
        #     assert (
        #         ai_message.ai_function_call is not None
        #         and ai_message.ai_function_call.arguments is not None
        #     )
        #     return get_current_weather(
        #         location=ai_message.ai_function_call.arguments.get("location"),
        #         unit=ai_message.ai_function_call.arguments.get("unit"),
        #     )

        events = [
            "What's the weather like in Boston, MA in farenheit? Make sure the location is qualified by the 2 letter state code."
        ]
        driver = driver_wrapper(
            events=events,
            system_message=MessageNode(
                role=Role.system,
                message_content="You are helpful assistant. Follow the instructions I give you. Do not respond until I ask you a question.",
            ),
            ai_driver=AIDriver(AltAI(model="gpt-3.5-turbo-16k"), event_manager),
            event_manager=event_manager,
            user_message_prompt_fn=lambda x: x.raw_event_str,
        )
        event, _ = next(driver)
        assert event.history
        self.assertEqual(len(event.history), 4)
        self.assertEqual(event.history[0].role, Role.system)
        self.assertEqual(
            event.history[0].message_content,
            "You are helpful assistant. Follow the instructions I give you. Do not respond until I ask you a question.",
        )
        self.assertEqual(event.history[1].role, Role.user)
        self.assertEqual(
            event.history[1].message_content,
            events[0],
        )

        # MessageNode's functions is set and explicit_function_call is also set
        fn_call_specs, _ = weather_event_function_call_param()

        assert event.history[1].functions == fn_call_specs
        self.assertEqual(
            event.history[1].explicit_fn_call,
            UserExplicitFunctionCall(name="get_current_weather"),
        )

        self.assertEqual(event.history[2].role, Role.assistant)
        assert event.history[2].ai_function_call
        self.assertEqual(
            event.history[2].ai_function_call.name, "get_current_weather"
        )
        self.assertEqual(event.history[2].message_content, "")
        self.assert_dicts_equal_for_some_keys(
            event.history[2].ai_function_call.model_dump()["arguments"],
            {"location": "Boston, MA", "unit": "fahrenheit", "temperature": 72},
            keys=["location", "unit"],
            keys_in_both=["temperature"],
        )

        self.assertEqual(event.history[3].role, Role.function)
        self.assertEqual(
            event.history[3].ai_function_call_result_name, "get_current_weather"
        )
        print(event.history[3].ai_function_call_result)

        self.assertEqual(
            event.event_obj,
            WeatherEvent(
                **{  # type: ignore[arg-type]
                    "location": "Boston, MA",
                    "temperature": int(event.event_obj.temperature),
                    "unit": "fahrenheit",
                }
            ),
        )

    @staticmethod
    def assert_dicts_equal_for_some_keys(dict1, dict2, keys, keys_in_both):
        """Asserts that the values of the given keys in the two dictionaries are equal.

        Args:
            dict1: The first dictionary.
            dict2: The second dictionary.
            keys: A list of keys to compare.
        """
        for key in keys_in_both:
            assert key in dict1
            assert key in dict2
        for key in keys:
            assert dict1[key] == dict2[key] or dict1[key] is dict2[key]


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSendToOpenAIAPI)
    runner = unittest.TextTestRunner()
    try:
        runner.run(suite)
    except Exception:  # pylint: disable=broad-except
        import pdb  # type: ignore

        pdb.post_mortem()
