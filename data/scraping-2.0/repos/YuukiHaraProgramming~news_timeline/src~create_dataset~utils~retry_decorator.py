import functools
import openai
import time
import sys

# Define the retry decorator
def retry_decorator(max_error_count=10, retry_delay=1): # Loop with a maximum of 10 attempts
    def decorator_retry(func):
        functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize an error count
            error_count = 0
            while error_count < max_error_count:
                try:
                    v = func(*args, **kwargs)
                    return v
                except openai.error.Timeout as e:
                    print("Timeout error occurred. Re-running the function.")
                    print(f"Timeout error: {e}")
                    error_count += 1
                except openai.error.APIError as e:
                    print("OPENAI API error occurred. Re-running the function.")
                    print(f"OPENAI API error: {e}")
                    error_count += 1
                except ValueError as e:
                    print("ValueError occurred. Re-running the function.")
                    print(f"ValueError: {e}")
                    error_count += 1
                except AttributeError as e: # For when other functions are called in function calling.
                    print(f"AttributeError occurred: {e}. Re-running the function.")
                    error_count += 1
                except TypeError as e: # For when other functions are called in function calling.
                    print("TypeError occurred. Re-running the function.")
                    print(f"TypeError: {e}")
                    error_count += 1
                except openai.error.InvalidRequestError as e:
                    print("InvalidRequestError occurred. Continuing the program.")
                    print(f"openai.error.InvalidRequestError: {e}")
                    break  # Exit the loop
                except Exception as e:
                    print("Exception error occurred. Re-running the function.")
                    print(f"Exeption: {e}")
                    error_count += 1
                time.sleep(retry_delay)  # If an error occurred, wait before retrying
            if error_count == max_error_count:
                sys.exit("Exceeded the maximum number of retries. Exiting the function.")
                return None
        return wrapper
    return decorator_retry