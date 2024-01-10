import json
from typing import Any, Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.sql_database.tool import ListSQLDatabaseTool
from pydantic.v1 import root_validator


class ListTableTool(ListSQLDatabaseTool):
    """Tool for getting table names."""

    name: str = "list_table_tool"
    description: str = f"""
- {name}:
  - Description: {name} can be used to list all available tables in the database.
  - Usage Schema: When involking {name}, ensure that you provide a JSON object adhering to the following schema:

    ```yaml
    ToolRequest:
      type: object
      properties:
        tool_name:
          type: string
          enum: ["{name}"]
      required: [tool_name]
    ```"""

    redis_url: str = "redis://localhost:6379"
    key_prefix: str = "sqlbot:tables:"
    client: Any = None
    # aclient: Any = None

    @root_validator(pre=True)
    def validate_environment(cls, values):
        from redis import Redis

        values["client"] = Redis.from_url(values["redis_url"], decode_responses=True)
        # TODO: async redis need to manually close the connection pool
        # from redis.asyncio import Redis as ARedis
        # values["aclient"] = ARedis.from_url(values["redis_url"], decode_responses=True)
        return values

    def _run(
        self,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for a specific table."""
        usable_tables = self.db.get_usable_table_names()
        return ", ".join(usable_tables)
        # res = {}
        # for key in self.client.scan_iter(f"{self.key_prefix}*"):
        #     table_name = key.removeprefix(self.key_prefix)
        #     if table_name in usable_tables:
        #         res[table_name] = self.client.get(key)
        # return json.dumps(res)
