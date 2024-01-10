from typing import Any, Dict
from bson import ObjectId
import openai
from tests.conftest_mock_openai import mock_completition_data
from tests.endpoints.inferences.inference_test_records import (
    DRAFT_PROMPT_VERSION_DB,
    LIVE_PROMPT_VERSION_DB,
    PROMPT_ID1,
    VALID_TEMPLATE,
)
from tests.shared_test_data import DEFAULT_RAW_COMPLETITION_REQUEST, ORG1, USER_BASIC
from tests.utils import TestInput, TestSpecList

PROMPT_ID2_NO_LIVE_VERSION = ObjectId("6468b05c1e5a37458856d10c")


FILLED_TEMPLATE = {
    "role": "system",
    "content": "a1: v1 a2: v2",
    "name": None,
}

VALID_REQ = {
    "prompt_version_id": str(LIVE_PROMPT_VERSION_DB["_id"]),
    "end_user_id": "u1",
    "source": "source1",
}

ORG2_NONE_ACCESS_KEYS = {
    **ORG1,
    "_id": ObjectId("bbbbbbbbbbbbbbbbbbbbbbbf"),
    "name": "org2",
    "access_keys": None,
}

USER_BASIC_ORG2_NO_ACCESS_KEY = {
    **USER_BASIC,
    "_id": ObjectId("aaaaaaaaaaaaaaaaaaaaaaaf"),
    "email": "noaccesskey@x.ai",
    "org_id": ORG2_NONE_ACCESS_KEYS["_id"],
}

LIVE_PROMPT_VERSION_DB2_ORG2 = {
    **LIVE_PROMPT_VERSION_DB,
    "_id": ObjectId("cccccccccccccccccccccccf"),
    "created_by_org_id": ORG2_NONE_ACCESS_KEYS["_id"],
}

# to test inference post by prompt id with multuple LIVE prompt versions
LIVE_PROMPT_VERSION_DB3_ORG1 = {
    **LIVE_PROMPT_VERSION_DB,
    "_id": ObjectId("aa6bae490e5a37458856aaaa"),
}

# to test inference post by prompt id with multuple LIVE prompt versions
LIVE_PROMPT_VERSION_DB4_ARCHIVED_ORG1 = {
    **LIVE_PROMPT_VERSION_DB,
    "prompt_id": PROMPT_ID2_NO_LIVE_VERSION,
    "status": "Archived",
    "_id": ObjectId("cc6bae490e5a37458856bbbb"),
}

LIVE_PROMPT_VERSION_DB5_TESTING_ORG1 = {
    **LIVE_PROMPT_VERSION_DB,
    "prompt_id": PROMPT_ID2_NO_LIVE_VERSION,
    "status": "Testing",
    "_id": ObjectId("cc6bae490e5a37458856cccc"),
}

test_db_data = {
    "users": [USER_BASIC, USER_BASIC_ORG2_NO_ACCESS_KEY],
    "orgs": [ORG1, ORG2_NONE_ACCESS_KEYS],
    "promptVersions": [
        LIVE_PROMPT_VERSION_DB,
        DRAFT_PROMPT_VERSION_DB,
        LIVE_PROMPT_VERSION_DB2_ORG2,
        LIVE_PROMPT_VERSION_DB3_ORG1,
        LIVE_PROMPT_VERSION_DB4_ARCHIVED_ORG1,
        LIVE_PROMPT_VERSION_DB5_TESTING_ORG1,
    ],
}

min_input: TestInput = {
    "request_body": {
        "prompt_version_id": str(LIVE_PROMPT_VERSION_DB["_id"]),
    }
}

all_input: TestInput = {
    "request_body": {
        "prompt_version_id": str(LIVE_PROMPT_VERSION_DB["_id"]),
        "end_user_id": "u1",
        "source": "s1",
        "client_ref_id": "u1-11",
        "request_timeout": 120.1,
        "template_args": {"arg1": "v1", "arg2": "v2"},
        "metadata": {"meta1": "m1"},
    }
}

expected_all_fields_head = {
    "created_by_user_id": USER_BASIC["_id"],
    "created_by_org_id": ORG1["_id"],
    "end_user_id": "u1",
    "source": "s1",
    "client_ref_id": "u1-11",
    "template_args": {"arg1": "v1", "arg2": "v2"},
    "prompt_id": PROMPT_ID1,
    "prompt_version_id": LIVE_PROMPT_VERSION_DB["_id"],
    "prompt_version_name": "prompt v1",
    "prompt_version_ids_considered": [],
    "status": "Processed",
    "metadata": {"meta1": "m1"},
    "request_timeout": 120.1,
    "request": {
        "provider": "OpenAI",
        "raw_request": {
            **DEFAULT_RAW_COMPLETITION_REQUEST,  # type: ignore[arg-type]
            **LIVE_PROMPT_VERSION_DB["model_config"],
            "messages": [FILLED_TEMPLATE],
            "user": "u1",
        },
    },
}

expected_min_fields_head = {
    "created_by_user_id": USER_BASIC["_id"],
    "created_by_org_id": ORG1["_id"],
    "template_args": {},
    "end_user_id": None,
    "source": None,
    "client_ref_id": None,
    "prompt_id": PROMPT_ID1,
    "prompt_version_id": LIVE_PROMPT_VERSION_DB["_id"],
    "prompt_version_ids_considered": [],
    "prompt_version_name": "prompt v1",
    "status": "Processed",
    "metadata": None,
    "request_timeout": None,
    "request": {
        "provider": "OpenAI",
        "raw_request": {
            **DEFAULT_RAW_COMPLETITION_REQUEST,  # type: ignore[arg-type]
            **LIVE_PROMPT_VERSION_DB["model_config"],
            "messages": [VALID_TEMPLATE],
        },
    },
}


def custom_db_validator_for_post_by_prompt_id(actual_db: Dict[str, Any]) -> bool:
    print(
        "\n custom_db_validator_for_post_by_prompt_id: \n"
        + f'<---- actual_db[prompt_version_id]: {actual_db["prompt_version_id"]} \n'
        + f'      actual_db[prompt_version_ids_considered]: {actual_db["prompt_version_ids_considered"]}\n'
        + f'      LIVE_PROMPT_VERSION_DB: {LIVE_PROMPT_VERSION_DB["_id"]}\n'
        + f'      LIVE_PROMPT_VERSION_DB3_ORG1: {LIVE_PROMPT_VERSION_DB3_ORG1["_id"]}\n'
    )
    assert len(actual_db["prompt_version_ids_considered"]) == 1
    assert actual_db["prompt_version_ids_considered"][0] in [
        LIVE_PROMPT_VERSION_DB["_id"],
        LIVE_PROMPT_VERSION_DB3_ORG1["_id"],
    ]

    assert actual_db["prompt_version_id"] in [
        LIVE_PROMPT_VERSION_DB["_id"],
        LIVE_PROMPT_VERSION_DB3_ORG1["_id"],
    ]

    return True


test_specs_post: TestSpecList = [
    #
    # Post by prompt_version_id
    #
    {
        "spec_id": "all input params",
        "mock_user": USER_BASIC,
        "input": all_input,
        "expected": {
            **expected_all_fields_head,
            "response": {
                "isError": False,
                "completition_duration_seconds": 6.6,  # value ignored in tests
                "is_client_connected_at_finish": True,
                "token_usage": mock_completition_data["usage"],
                "raw_response": mock_completition_data,
            },
        },
    },
    {
        "spec_id": "min input params",
        "mock_user": USER_BASIC,
        "input": min_input,
        "expected": {
            **expected_min_fields_head,
            "prompt_version_ids_considered": [],
            "response": {
                "isError": False,
                "completition_duration_seconds": 6.6,  # value ignored in tests
                "is_client_connected_at_finish": True,
                "token_usage": mock_completition_data["usage"],
                "raw_response": mock_completition_data,
            },
        },
    },
    #
    # Post by prompt_id
    #
    {
        "spec_id": "post by prompt id",
        "mock_user": USER_BASIC,
        "input": {"request_body": {"prompt_id": str(PROMPT_ID1)}},
        "expected": {
            "this_is": "ignored by post test generator currently bc of custom db validator fn in  expected_db "
        },
        "expected_db": custom_db_validator_for_post_by_prompt_id,
    },
    {
        "spec_id": "no live prompt_version for post by prompt_id",
        "mock_user": USER_BASIC,
        "input": {"request_body": {"prompt_id": str(PROMPT_ID2_NO_LIVE_VERSION)}},
        "expected": 404,
    },
    #
    # Permission tests
    #
    {
        "spec_id": "user shouldn't inference other orgs promptversion",
        "mock_user": USER_BASIC,
        "input": {
            "request_body": {
                "prompt_version_id": str(LIVE_PROMPT_VERSION_DB2_ORG2["_id"])
            }
        },
        "expected": 404,
    },
    {
        "spec_id": "no org token",
        "mock_user": USER_BASIC_ORG2_NO_ACCESS_KEY,
        "input": {
            "request_body": {"prompt_version_id": str(PROMPT_ID2_NO_LIVE_VERSION)}
        },
        "expected": 400,
    },
    #  request  validations
    #
    {
        "spec_id": "both prompt_version_id and prompt_id",
        "mock_user": USER_BASIC,
        "input": {"request_body": {**VALID_REQ, "prompt_id": str(PROMPT_ID1)}},
        "expected": 422,
    },
    {
        "spec_id": "invalid prompt_version_id",
        "mock_user": USER_BASIC,
        "input": {"request_body": {**VALID_REQ, "prompt_version_id": "xxxx"}},
        "expected": 422,
    },
    {
        "spec_id": "non existent prompt_version_id",
        "mock_user": USER_BASIC,
        "input": {
            "request_body": {
                **VALID_REQ,
                "prompt_version_id": ObjectId("ffffffffffffffffffffffff"),
            }
        },
        "expected": 404,
    },
    {
        "spec_id": "should not inference on draft promptVersion",
        "mock_user": USER_BASIC,
        "input": {
            "request_body": {
                **VALID_REQ,
                "prompt_version_id": str(DRAFT_PROMPT_VERSION_DB["_id"]),
            }
        },
        "expected": 422,
    },
    {
        "spec_id": "invalid extra field",
        "mock_user": USER_BASIC,
        "input": {"request_body": {**VALID_REQ, "foo": "moo"}},
        "expected": 422,
    },
]

post_openai_error_test_specs: TestSpecList = [
    {
        "spec_id": "mock openai apiError",
        "mock_user": USER_BASIC,
        "input": all_input,
        "mock_exception": openai.APIError("mocked error"),
        "expected": {
            "detail": {
                "message": "OpenAI API Error",
                "openAI_error_class": "openai.error.APIError",
                "openAI_message": "mocked error",
                "openAI_error": None,
            }
        },
        "expected_db": {
            **expected_all_fields_head,
            "status": "CompletitionError",
            "response": {
                "isError": True,
                "completition_duration_seconds": 6.6,
                "is_client_connected_at_finish": True,
                "error": {
                    "error_class": "openai.error.APIError",
                    "message": "mocked error",
                },
            },
        },
    },
    {
        "spec_id": "mock openai timeout",
        "mock_user": USER_BASIC,
        "input": min_input,
        "mock_exception": openai.error.Timeout(  # pyright: ignore[reportGeneralTypeIssues]
            "mocked timeout"
        ),
        "expected": {
            "detail": {
                "message": "OpenAI API Timeout Error",
                "openAI_error_class": "openai.error.Timeout",
                "openAI_message": "mocked timeout",
                "openAI_error": None,
            }
        },
        "expected_db": {
            **expected_min_fields_head,
            "status": "CompletitionTimeout",
            "response": {
                "isError": True,
                "completition_duration_seconds": 6.6,  # value ignored in tests
                "is_client_connected_at_finish": True,
                "error": {
                    "error_class": "openai.error.Timeout",
                    "message": "mocked timeout",
                },
            },
        },
    },
]
