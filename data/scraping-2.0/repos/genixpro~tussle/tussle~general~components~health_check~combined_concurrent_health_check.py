import datetime
from .db_access_health_check import DbAccessHealthCheck
from .openai_integration_health_check import OpenAIIntegrationHealthCheck
from tussle.general.logger import get_logger
import concurrent.futures
from .health_check_base import HealthCheckBase
import traceback
import json
import dataclasses


@dataclasses.dataclass
class HealthCheckConfiguration:
    name: str
    check_function: callable
    required: bool


class CombinedConcurrentHealthCheck(HealthCheckBase):
    """
    This combines several other health checks and runs them concurrently
    """

    def __init__(self):
        super().__init__()
        self.health_check_timeout = 15

        self.health_checks = [
            HealthCheckConfiguration(
                name="db_access",
                check_function=DbAccessHealthCheck().check,
                required=True
            ),
            HealthCheckConfiguration(
                name="openai_integration",
                check_function=OpenAIIntegrationHealthCheck().check,
                required=True
            ),
        ]

        self.health_checks_by_name = {
            health_check.name: health_check for health_check in self.health_checks
        }
        self.logger = get_logger("CombinedConcurrentHealthCheck")
        self.thread_executor = concurrent.futures.ThreadPoolExecutor()

    def check(self):
        health_check_results = {}

        try:
            health_check_futures = {}
            for health_check_config in self.health_checks:
                health_check_futures[health_check_config.name] = self.thread_executor.submit(health_check_config.check_function)

            target_finish_time = datetime.datetime.now() + datetime.timedelta(seconds=self.health_check_timeout)

            for name, future in health_check_futures.items():
                time_remaining = max(0.01, (target_finish_time - datetime.datetime.now()).total_seconds())
                is_required = self.health_checks_by_name[name].required
                try:
                    check_result = future.result(timeout=time_remaining)
                    check_result['required'] = is_required
                    health_check_results[name] = check_result
                except concurrent.futures.TimeoutError:
                    self.logger.error(f"Health check {name} timed out after {self.health_check_timeout} seconds.")
                    health_check_results[name] = {
                        "healthy": False,
                        'required': is_required,
                        "details": {
                            "error": f"Health check {name} timed out after {self.health_check_timeout} seconds."
                        }
                    }

            required_health_check_results = [
                health_check_results[health_check_config.name] for health_check_config in self.health_checks
                if health_check_config.required
            ]

            healthy = all([result['healthy'] for result in required_health_check_results])

            result_str = json.dumps(health_check_results, indent=4)

            self.logger.info(f"Combined concurrent health check result:\nHealthy: {healthy}. Details:\n{result_str}")

            return {
                "healthy": healthy,
                "details": health_check_results,
            }
        except Exception as e:
            self.logger.error(f"Error while checking chart processing:\n{traceback.format_exc()}")
            return {
                "healthy": False,
                "details": health_check_results,
                "error": traceback.format_exc()
            }
