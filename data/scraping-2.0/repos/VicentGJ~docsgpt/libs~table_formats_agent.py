"""Agent for working with table format."""
from typing import Any, List, Optional, Union

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.base_language import BaseLanguageModel


def create_table_format_agent(
    llm: BaseLanguageModel,
    path: Union[str],
    extension: str,
    pandas_kwargs: Optional[dict] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Create csv agent by loading to a dataframe and using pandas agent."""
    try:
        import pandas as pd
    except ImportError:
        raise ValueError(
            "pandas package not found, please install with `pip install pandas`"
        )
    

    _kwargs = pandas_kwargs or {}
    if isinstance(path, str):
        if extension == 'csv':
            df = pd.read_csv(path, **_kwargs)
        elif extension == 'xlsx':
            df = pd.read_excel(path, **_kwargs)
        elif extension == 'ods':
            df = pd.read_excel(path, **_kwargs,engine='odf')
        else:
            raise ValueError(f"Incorrect extension file")
    
    else:
        raise ValueError(f"Expected str, got {type(path)}")
    return create_pandas_dataframe_agent(llm, df, **kwargs)