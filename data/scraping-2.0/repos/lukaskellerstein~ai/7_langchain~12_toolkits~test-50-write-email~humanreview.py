from langchain.callbacks.human import HumanApprovalCallbackHandler


# ---------------------------
# HUMAN REVIEW
# ---------------------------
def _should_check(serialized_obj: dict) -> bool:
    print(serialized_obj)
    return (
        serialized_obj.get("name") == "send_gmail_message"
        or serialized_obj.get("name") == "create_gmail_draft"
    )


def _approve(_input: str) -> bool:
    msg = (
        "Do you approve of the following input? "
        "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no."
    )
    msg += "\n\n" + _input + "\n"
    resp = input(msg)
    return resp.lower() in ("yes", "y")


callbacks = [HumanApprovalCallbackHandler(should_check=_should_check, approve=_approve)]
