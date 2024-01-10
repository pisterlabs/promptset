from langchain.tools import tool

class CalculatorTools:
    @tool("Perform calculation")
    def calculate(self, expression):
        """
        Performs a calculation based on the given mathematical expression.
        :param expression: A string containing the mathematical expression to be calculated.
        :return: The result of the calculation.
        """
        try:
            # Use a safer, simpler method for calculation
            result = eval(expression, {"__builtins__": None}, {})
            return result
        except Exception as e:
            return f"Error in calculation: {str(e)}"
