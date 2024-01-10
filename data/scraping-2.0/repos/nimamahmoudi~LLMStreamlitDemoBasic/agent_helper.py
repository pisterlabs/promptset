# from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler
from tenacity import retry, wait_exponential, stop_after_attempt

def bind_logger(toolClass):
    class newToolClass(toolClass):
        def __init__(self, tool_name: str, st_cb: StreamlitCallbackHandler, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.st_cb = st_cb
            self.tool_name = tool_name

        def run(self, *args, **kwargs):
            print(f"Running {toolClass.__name__} {[*args]}, {kwargs}")

            if self.st_cb._current_thought is None:
                self.st_cb.on_llm_start({}, [])

            args_str = ' '.join(args) + ' ' + ' '.join([f'{k}=`{v}`' for k, v in kwargs.items()])
            self.st_cb.on_tool_start({'name': self.tool_name}, args_str)

            try:
                ret_val = retry(
                    wait=wait_exponential(min=2, max=20),
                    stop=stop_after_attempt(5),
                )(super().run)(*args, **kwargs)
                self.st_cb.on_tool_end(ret_val)
                return ret_val
            except Exception as e:
                original_exception = e.last_attempt.result()
                print(f"Exception {original_exception} in {toolClass.__name__} {[*args]}, {kwargs}")
                raise original_exception
            
        
    return newToolClass
        
from functools import wraps

def retry_and_streamlit_callback(st_cb: StreamlitCallbackHandler, tool_name: str):
    if st_cb is None:
        return lambda x: x

    def decorator(tool_func):
        @wraps(tool_func)
        def decorated_func(*args, **kwargs):
            print(f"Running {tool_name} {args}, {kwargs}")

            if st_cb._current_thought is None:
                st_cb.on_llm_start({}, [])

            args_str = ' '.join(args) + ' ' + ' '.join([f'{k}=`{v}`' for k, v in kwargs.items()])
            st_cb.on_tool_start({'name': tool_name}, args_str)

            @retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(5))
            def retry_wrapper():
                return tool_func(*args, **kwargs)

            try:
                ret_val = retry_wrapper()
                st_cb.on_tool_end(ret_val)
                return ret_val
            except Exception as e:
                print(f"Exception {e} in {tool_name} {args}, {kwargs}")
                raise e

        return decorated_func

    return decorator
