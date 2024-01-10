""" Logging module for ba_timeseries_gradients.

This module provides a logger for the openai_api_wrapper package.
The logger is imported into other modules using the following snippet:
```python
import logging
from ba_timeseries_gradients import logs

LOGGER_NAME = logs.LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)
```
"""
import logging

from openai_api_wrapper import constants

LOGGER_NAME = constants.LOGGER_NAME
logger = logging.getLogger(LOGGER_NAME)

cf = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
cf.setFormatter(formatter)
logger.addHandler(cf)
