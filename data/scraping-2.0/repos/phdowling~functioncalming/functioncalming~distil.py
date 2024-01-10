import inspect
from pydantic import BaseModel, RootModel, create_model


def distillery(function_name: str | None = None, function_description: str | None = None):
    """
    Tag a function as a functioncalming distillery for distillation training data generation.

    The training data that functioncalming will log for uses of this function will be altered as follows:
     - The original system prompt of the run will be replaced by the new system prompt given in the decorator
     - The message history will use the **return type** of this function as a tool, not the input signature
     - The message history will be re-written such called the function generated from the return type,
        producing the actual object returned by the function call directly

    For example, let's look at this (overly simplified) example of a refinement pipeline:
    ```
    import datetime
    from pydantic import BaseModel
    from functioncalming import distillery, get_completion

    class Event(BaseModel):
        name: str
        description: str
        date_time: datetime.datetime | None = None

    @distillery("Fully extract the given text into an event, making sure to fill in all inferable fields.")
    async def refine_event(
        event: Event,
        user_message: str  # the initial user message (magic param that is hidden from OpenAI model internally)
    ) -> Event:
        messages = [
            {
                "role": "system",
                "content": "Given some unstructured data and a partially extracted object, complete any missing fields."
            }
            {"role": "user", "content": user_message}
            {"role": "system", "content": f"Partial extraction: {event.model_dump_json()}\n. Please make improvements."}
        ]
        refined_event = (await get_completion(messages=messages, tools=[Event], ...))[0][0]
        return refined_event

    # non-distilled two-step pipeline
    refined_event = await get_completion(
        system_prompt="Extract the given data into an event. "
                      "Focus only fill name and description for now, "
                      "do not fill date information (the refine function will do this automatically)",
        user_message="... some unstructured event data ...",
        tools=[refine_event],
        log_finetuning_to="event_finetune.jsonl"
    )
    ```
    In the final logged finetuning data, the call to refine_event will be hidden and replaced by a call to the new
    function name. The message history will show the model directly making a call to Event(...) with the final
    refinement result.

    :param function_name: The new tool/function name to use in the adjusted message history
    :param function_description: The new tool/function description to use in the adjusted message history
    :return: the function (unchanged, this decorator just attaches some metadata that functioncalming uses internally)
    """
    def _distillery(fun):
        return_type = inspect.signature(fun).return_annotation
        if return_type is None:
            raise TypeError("@distillery() functions must have a return type annotation that is not None")

        if not (isinstance(return_type, type) and issubclass(return_type, (BaseModel, RootModel))):
            if function_name is None:
                raise ValueError(
                    "If return type of a distillery is not a BaseModel, you need to supply a function_name"
                )

            try:
                use_return_type = RootModel[return_type]
                use_return_type.__name__ = function_name
                use_return_type.__doc__ = function_description
            except Exception as e:
                raise TypeError(
                    "@distillery() function return value must be BaseModel or RootModel-compatible "
                    "and declared them in their return annotation"
                ) from e
        else:
            use_return_type = create_model(
                function_name or return_type.__name__,
                __base__=return_type,
                __doc__=function_description,
            )

        fun.__is_functioncalming_distillery__ = True
        fun.__functioncalming_distil_model__ = use_return_type
        return fun
    return _distillery
