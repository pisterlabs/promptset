from typing import Dict, Any, Optional

from xpuls.mlmonitor.langchain.patches import patch_run
from xpuls.mlmonitor.langchain.patches.patch_invoke import patch_invoke
from xpuls.mlmonitor.langchain.profiling.prometheus import LangchainPrometheusMetrics
from xpuls.mlmonitor.langchain.xpuls_client import XpulsAILangChainClient


class LangchainTelemetry:
    def __init__(self, default_labels: Dict[str, Any],
                 enable_prometheus: bool = True,):
        self.ln_metrics = LangchainPrometheusMetrics(default_labels)

        self.xpuls_client = XpulsAILangChainClient()

        self.default_labels = default_labels
        self.enable_prometheus = enable_prometheus

    def auto_instrument(self):
        patch_run(self.ln_metrics, self.xpuls_client)
        patch_invoke(self.ln_metrics, self.xpuls_client)
        print("** ProfileML -> Langchain auto-instrumentation completed successfully **")
