# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor
# import traceback

# def apply_parallel(df, func, max_workers=20, max_retries=5):
#     def process_wrapper(row):
#         for attempt in range(max_retries):
#             try:
#                 # Call the processing function and return the result
#                 return func(row[1]), None  # row[1] is the row data in iterrows()
#             except Exception as e:
#                 error_msg = traceback.format_exc()
#                 if attempt < max_retries - 1:
#                     continue
#                 return [], error_msg
#         return [], 'Unknown error'  # Fallback error message

#     results = []
#     errors = {}

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         for row_index, (result, error) in zip(df.index, executor.map(process_wrapper, df.iterrows())):
#             results.extend(result)
#             if error:
#                 print(f"Error in row {row_index}: {error}")
#                 errors[row_index] = error

#     # Create result DataFrame
#     result_df = pd.DataFrame(results)

#     # Create status DataFrame
#     status_df = df.copy()
#     status_df['is_successful'] = ~status_df.index.isin(errors.keys())
#     status_df['error_msg'] = status_df.index.map(errors.get)

#     return result_df, status_df


# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor
# import traceback
# import time
# import openai  # Assuming this is the library you are using that throws openai.RateLimitError
# from tqdm import tqdm  # Make sure to install tqdm if not already installed

# def apply_parallel(df, func, max_workers=20, max_retries=5):
#     def process_wrapper(row):
#         start_time = time.time()
#         while True:
#             try:
#                 # Call the processing function and return the result
#                 return func(row[1]), None  # row[1] is the row data in iterrows()
#             except openai.RateLimitError:
#                 # If RateLimitError, wait and retry without counting it as a failure
#                 if time.time() - start_time < 600:  # 10 minutes
#                     time.sleep(30)  # Wait for 30 seconds before retrying
#                     continue
#                 else:
#                     return [], 'Rate limit exceeded for 10 minutes'
#             except Exception as e:
#                 error_msg = traceback.format_exc()
#                 if attempt < max_retries - 1:
#                     attempt += 1
#                     continue
#                 return [], error_msg
#         return [], 'Unknown error'  # Fallback error message

#     results = []
#     errors = {}

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         for row_index, (result, error) in zip(df.index, tqdm(executor.map(process_wrapper, df.iterrows()), total=len(df))):
#             results.extend(result)
#             if error:
#                 errors[row_index] = error

#     # Create result DataFrame
#     result_df = pd.DataFrame(results)

#     # Create status DataFrame
#     status_df = df.copy()
#     status_df['is_successful'] = ~status_df.index.isin(errors.keys())
#     status_df['error_msg'] = status_df.index.map(errors.get)

#     return result_df, status_df




import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import time
import openai  # Assuming this is the library you are using that throws openai.RateLimitError
from tqdm import tqdm  # Make sure to install tqdm if not already installed

def apply_parallel(df, func, max_workers=20, max_retries=5):
    def process_wrapper(row_index, row_data):
        start_time = time.time()
        attempt = 0
        while True:
            # check for the openai version:
            openai_version = openai.__version__
            # if openai version starts with 1.  
            if openai_version.startswith("1."):
                try:
                    # Call the processing function and return the result
                    return row_index, func(row_data), None  # Returning row_index for tracking
                except openai.RateLimitError:
                    # If RateLimitError, wait and retry without counting it as a failure
                    if time.time() - start_time < 600:  # 10 minutes
                        time.sleep(30)  # Wait for 30 seconds before retrying
                        continue
                    else:
                        return row_index, [], 'Rate limit exceeded for 10 minutes'
                except Exception as e:
                    error_msg = traceback.format_exc()
                    if attempt < max_retries - 1:
                        attempt += 1
                        continue
                    return row_index, [], error_msg
            else:
                try:
                    # Call the processing function and return the result
                    return row_index, func(row_data), None  # Returning row_index for tracking
                except openai.error.RateLimitError:
                    # If RateLimitError, wait and retry without counting it as a failure
                    if time.time() - start_time < 600:  # 10 minutes
                        time.sleep(30)  # Wait for 30 seconds before retrying
                        continue
                    else:
                        return row_index, [], 'Rate limit exceeded for 10 minutes'
                except Exception as e:
                    error_msg = traceback.format_exc()
                    if attempt < max_retries - 1:
                        attempt += 1
                        continue
                    return row_index, [], error_msg
        return row_index, [], 'Unknown error'  # Fallback error message

    results = []
    errors = {}
    success_count = 0
    failure_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_wrapper, idx, row): idx for idx, row in df.iterrows()}

        # Progress bar setup
        pbar = tqdm(total=len(df), desc="Processing rows")

        # As each future completes
        for future in as_completed(futures):
            row_index, result, error = future.result()
            results.extend(result)
            if error:
                errors[row_index] = error
                failure_count += 1
                pbar.set_description(f"Processing rows (Success: {success_count}, Failures: {failure_count})")
            else:
                success_count += 1
                pbar.set_description(f"Processing rows (Success: {success_count}, Failures: {failure_count})")
            pbar.update(1)

        pbar.close()

    # Create result DataFrame
    result_df = pd.DataFrame(results)

    # Create status DataFrame
    status_df = df.copy()
    status_df['is_successful'] = ~status_df.index.isin(errors.keys())
    status_df['error_msg'] = status_df.index.map(errors.get)

    return result_df, status_df
