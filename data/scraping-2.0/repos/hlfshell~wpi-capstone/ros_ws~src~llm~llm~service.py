from typing import Optional

import rclpy
from capstone_interfaces.srv import LLM
from llm.providers import OpenAI, PaLM
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node


class LLMService(Node):
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        super().__init__("llm_service")

        self.declare_parameter(
            "provider", "", ParameterDescriptor(description="LLM service provider")
        )
        self.declare_parameter(
            "model", "", ParameterDescriptor(description="LLM service model")
        )

        if provider is None:
            provider = self.get_parameter("provider").get_parameter_value().string_value
        if model is None:
            model = self.get_parameter("model").get_parameter_value().string_value

        if provider == "openai":
            self.provider = OpenAI(None, model)
        elif provider == "palm":
            self.provider = PaLM(None, model)
        else:
            raise ValueError(f"Unknown provider {provider}")

        self.srv = self.create_service(LLM, "prompt", self.prompt_callback)

    def prompt_callback(self, request: LLM.Request, response: LLM.Response):
        self.get_logger().debug(request.prompt)
        self.get_logger().debug(request.temperature)

        result = self.provider.prompt(request.prompt, request.temperature)

        self.get_logger().debug(result)

        response.result = result

        return response


def main(args=None):
    # initialize the ROS communication
    rclpy.init(args=args)
    # declare the node constructor
    service = LLMService()
    # pause the program execution, waits for a request to kill the node (ctrl+c)
    rclpy.spin(service)
    # shutdown the ROS communication
    rclpy.shutdown()


if __name__ == "__main__":
    main()
