from langflow.interface.custom.custom_component import CustomComponent
from langchain.schema import Document


class FlowRunner(CustomComponent):
    display_name = "Flow Runner"

    def build_config(self):
        flows = self.list_flows()
        names = [flow.name for flow in flows]
        return {
            "flow_name": {"is_list": True, "options": names},
            "inputs": {"input_types": ["str"]},
        }

    def build(self, flow_name: str, inputs: str) -> Document:
        """
        Run a flow with the given inputs
        """
        flows = self.list_flows()
        flow = next((f for f in flows if f.name == flow_name), None)
        if flow is None:
            raise ValueError(f"Flow {flow_name} not found")
        flow = self.load_flow(flow_name)
        return flow(inputs)
