import pandas as pd
import concurrent.futures
from langchain.schema.output_parser import StrOutputParser
from utils import save_tmp_file
from utils_multiline_table import df_to_multiline_table, parse_multiline_table
from data_handling import chunk_dataframe, get_input_columns, get_output_column_name
from langchain_core.prompts import PromptTemplate


class EvaluateAgainstDataset:
    def __init__(
        self,
        model,
        df_original,
        max_chunk_rows,
        concurrency=3,
    ):
        """
        Initialize the LangChainEvaluator class.

        Args:
        model: LangChain model used for generating data.
        prompt_template: Template for the prompt to use with LangChain.
        df_original: The original DataFrame for comparison.
        attempt_no: Identifier for the prompt being tested.
        plan_id: Identifier for the ToT (Tree of Thought plan) being tested.
        max_chunk_rows: Maximum number of rows per chunk for processing.
        concurrency: Number of concurrent threads for processing. Default is 3.
        """
        self.model = model
        self.df_original = df_original
        self.max_chunk_rows = max_chunk_rows
        self.concurrency = concurrency

        self.prompt_template = None
        self.attempt_no = None
        self.plan_id = None

    def get_chain(self):
        """
        Returns the LangChain chain.
        """
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def calculate_accuracy(self, df_generated, output_column_name):
        """
        Calculate the accuracy of generated data compared to the original dataset.

        Args:
        df_generated (DataFrame): The DataFrame generated from LangChain.
        output_column_name (str): The name of the output column to compare.

        Returns:
        float: The accuracy percentage.
        """
        # Add the 'Truth' column to the generated DataFrame
        df_generated = df_generated.assign(Truth=self.df_original[output_column_name])

        if pd.api.types.is_numeric_dtype(df_generated["Truth"]):
            # Cast the output column to numeric if 'Truth' is numeric
            df_generated[output_column_name] = pd.to_numeric(
                df_generated[output_column_name], errors="coerce"
            )
            df_generated["Is Correct?"] = df_generated.apply(
                lambda row: abs(row[output_column_name] - row["Truth"]) <= 2, axis=1
            )
        else:
            df_generated["Is Correct?"] = (
                df_generated["Truth"].str.lower()
                == df_generated[output_column_name].str.lower()
            )

        # Calculate the accuracy
        accuracy = df_generated["Is Correct?"].sum() / len(df_generated) * 100

        return accuracy, df_generated

    def add_input_columns_to_df(self, df_original, df_generated, input_columns):
        """
        Adds the input columns from the original DataFrame to the generated DataFrame.
        """
        # Add columns with 'input' in their name from df_original to df_generated
        for column in input_columns:
            df_generated[column] = df_original[column]
        return df_generated

    def move_input_columns(self, df_generated, input_columns):
        """
        Moves the input columns right after the 'ROW_NO' field in the DataFrame.
        """
        # Move the input columns right after the 'ROW_NO' field
        for column in reversed(input_columns):  # reverse to keep the original order
            df_generated.insert(
                df_generated.columns.get_loc("ROW_NO") + 1,
                column,
                df_generated.pop(column),
            )
        return df_generated

    def calculate_and_print_accuracy(self, df, output_column_name):
        """
        Calculates the accuracy of the generated data for the entire dataset and prints the result.
        """
        # Calculate the accuracy of the generated data for the entire dataset
        accuracy, df = self.calculate_accuracy(df, output_column_name)
        print(f"Correct answers: {accuracy:.2f}%")
        return accuracy, df

    def invoke(self, prompt_str, plan_id, attempt_no):
        """
        Invokes the test prompt against the original dataset.

        Returns:
        DataFrame: Generated DataFrame with results.
        float: Accuracy percentage.
        """
        self.prompt_template = PromptTemplate.from_template(prompt_str)
        self.plan_id = plan_id
        self.attempt_no = attempt_no

        # Split the dataset into chunks
        df_chunks = chunk_dataframe(self.df_original, self.max_chunk_rows)

        # Process each chunk and aggregate the results
        df_generated = self.process_chunks_and_aggregate(df_chunks)

        # Get input columns and add them to the generated DataFrame
        input_columns = get_input_columns(self.df_original)
        df_generated = self.add_input_columns_to_df(
            self.df_original, df_generated, input_columns
        )

        # Move the input columns right after the 'ROW_NO' field
        df_generated = self.move_input_columns(df_generated, input_columns)

        # Find the name of the 'OUTPUT' column
        output_column_name = get_output_column_name(df_generated)

        # Calculate the accuracy and print the result
        accuracy, df_generated = self.calculate_and_print_accuracy(
            df_generated, output_column_name
        )

        return df_generated, accuracy

    def process_chunks_and_aggregate(self, df_chunks):
        """
        Process the chunks of DataFrame and aggregate the results.

        Args:
        df_chunks: List of DataFrame chunks.

        Returns:
        DataFrame: Aggregated DataFrame of generated results.
        """
        # Initialize an ordered dictionary to store results from each chunk
        chunk_list = []

        # Define the number of workers to use (maximum of `concurrency`)
        num_workers = min(len(df_chunks), self.concurrency)

        # Use a ThreadPoolExecutor to execute multiple chunks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.process_chunk_with_retry, j, chunk): j
                for j, chunk in enumerate(df_chunks, start=1)
            }

            # Get the results from the completed Future instances
            for future in concurrent.futures.as_completed(futures):
                chunk_list.append(future.result())

        # Concatenate the results from each chunk into a single DataFrame and sort
        df_generated = pd.concat(chunk_list)
        # convert ROW_NO to int
        df_generated["ROW_NO"] = df_generated["ROW_NO"].astype(int)
        df_generated = df_generated.sort_values(by=["ROW_NO"]).reset_index(drop=True)

        return df_generated

    def process_chunk_with_retry(self, j, chunk, retries=2):
        """
        Process a chunk of DataFrame with retries in case of failure.

        Args:
        j: Chunk identifier.
        chunk: DataFrame chunk to be processed.
        retries: Number of retry attempts. Default is 3.

        Raises:
        ValueError: If processing fails after specified retries.

        Returns:
        DataFrame: DataFrame of generated results for the chunk.
        """
        for retry in range(retries):
            try:
                return self.process_chunk(j, chunk, retry)
            except Exception as e:
                print("Retrying...", e)

        raise ValueError(f"Failed to process chunk {j} after {retries} retries")

    def process_chunk(self, j, chunk, retry=0):
        """
        Process a single chunk of DataFrame.

        Args:
        j: Chunk identifier.
        chunk: DataFrame chunk to be processed.
        retry: Current retry attempt.

        Returns:
        DataFrame: DataFrame of generated results for the chunk.
        """
        if self.prompt_template is None:
            raise ValueError("Prompt template is not set!")

        file_prefix = (
            f"03-plan-{self.plan_id}-attempt-{self.attempt_no}-chunk-{j}-retry-{retry}"
        )

        print(f"Getting chunk {j} retry {retry} with {len(chunk)} rows...", flush=True)
        prompt_formatted = self.prompt_template.format(
            input_table=df_to_multiline_table(chunk, is_remove_output_field=True)
        )
        save_tmp_file(
            f"{file_prefix}-(1)-request.md",
            prompt_formatted,
        )

        # spaces as many retry times
        retry_spaces = " " * retry

        answer_prompt_gen_chunk = self.get_chain().invoke(
            {
                "input_table": df_to_multiline_table(chunk, is_remove_output_field=True)
                + retry_spaces
            }
        )
        save_tmp_file(
            f"{file_prefix}-(2)-response.md",
            answer_prompt_gen_chunk,
        )

        df_generated_chunk = parse_multiline_table(
            answer_prompt_gen_chunk, expected_count=len(chunk)
        )

        return df_generated_chunk
