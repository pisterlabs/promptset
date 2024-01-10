from greptimeai.extractor.openai_extractor import OpenaiExtractor


def test_is_stream():
    assert OpenaiExtractor.is_stream() is False
    assert OpenaiExtractor.is_stream(stream=None) is False
    assert OpenaiExtractor.is_stream(stream=True) is True
    assert OpenaiExtractor.is_stream(stream=False) is False

    assert OpenaiExtractor.is_stream(stream="True") is True
    assert OpenaiExtractor.is_stream(stream="False") is False


def test_get_trace_info():
    # Test case 1: No trace_id or span_id provided
    result = OpenaiExtractor.get_trace_info()
    assert result is None

    # Test case 2: Only trace_id provided
    result = OpenaiExtractor.get_trace_info(trace_id="12345")
    assert result == ("12345", "")

    # Test case 3: Only span_id provided
    result = OpenaiExtractor.get_trace_info(span_id="67890")
    assert result == ("", "67890")

    # Test case 4: Both trace_id and span_id provided
    result = OpenaiExtractor.get_trace_info(trace_id="12345", span_id="67890")
    assert result == ("12345", "67890")

    # Test case 5: Extra headers with trace_id and span_id provided
    extra_headers = {"x-trace-id": "12345", "x-span-id": "67890"}
    result = OpenaiExtractor.get_trace_info(extra_headers=extra_headers)
    assert result == ("12345", "67890")

    # Test case 6: Extra headers with no trace_id or span_id provided
    extra_headers = {"x-trace-id": "", "x-span-id": ""}
    result = OpenaiExtractor.get_trace_info(extra_headers=extra_headers)
    assert result is None

    # Test case 6: trace_id, span_id and Extra headers all provided
    extra_headers = {"x-trace-id": "x-12345", "x-span-id": "x-67890"}
    result = OpenaiExtractor.get_trace_info(
        trace_id="12345", span_id="67890", extra_headers=extra_headers
    )
    assert result == ("x-12345", "x-67890")


def test_get_user_id():
    # Test case 1: No user_id provided
    result = OpenaiExtractor.get_user_id()
    assert result is None

    # Test case 2: Only user_id provided
    result = OpenaiExtractor.get_user_id(user_id="12345")
    assert result == "12345"

    # Test case 3: Extra headers with user_id provided
    extra_headers = {"x-user-id": "12345"}
    result = OpenaiExtractor.get_user_id(extra_headers=extra_headers)
    assert result == "12345"

    # Test case 4: Extra headers with no user_id provided
    extra_headers = {}
    result = OpenaiExtractor.get_user_id(extra_headers=extra_headers)
    assert result is None

    # Test case 5: user_id and Extra headers all provided
    extra_headers = {"x-user-id": "x-12345"}
    result = OpenaiExtractor.get_user_id(user_id="12345", extra_headers=extra_headers)
    assert result == "x-12345"

    # Test case 6: user and user_id both provided
    result = OpenaiExtractor.get_user_id(user="user", user_id="12345")
    assert result == "12345"

    # Test case 7: user and user_id and Extra headers all provided
    extra_headers = {"x-user-id": "x-12345"}
    result = OpenaiExtractor.get_user_id(
        user="user", user_id="12345", extra_headers=extra_headers
    )
    assert result == "x-12345"
