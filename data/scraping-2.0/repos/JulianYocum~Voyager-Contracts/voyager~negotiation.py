import openai
import logging
from api_keys import openai_api_key
from voyager.prompts import load_prompt

class Negotiator:
    def __init__(self, name, task, other_name, other_task, scenario, model="gpt-3.5-turbo", temperature=0.1):
        self.name = name
        self.task = task
        self.other_name = other_name
        self.other_task = other_task
        self.scenario = scenario
        self.model = model
        self.temperature = temperature

        openai.api_key = openai_api_key

        # Including both tasks in the system prompt
        system_prompt = load_prompt("negotiator")
        self.system_prompt = f"Your name is {name}\n\nYour Task: {task}\n\nOther Agent's Name: {other_name}\n\nOther Agent's Task: {other_task}\n\nScenario: {scenario}\n\n{system_prompt}"

        self.reset()
        

    def reset(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def generate_message(self, content=None):
        if content: 
            self.messages.append({"role": "user", "content": content})
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
        )
        
        # Parsing the response based on the new structure
        split_response = response['choices'][0]['message']['content'].split('[message]')
        inner_thought = split_response[0].replace('[thinking]', '').strip()
        message = split_response[1].strip() if len(split_response) > 1 else ""
        
        self.messages.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
        
        return inner_thought, message

class Negotiation:
    def __init__(self, agent1, agent2, max_turns=6, save_dir='logs'):
        self.agent1 = agent1
        self.agent2 = agent2
        self.max_turns = max_turns
        self.save_dir = save_dir
        self.reset()
        self.logger = self.setup_custom_logger()
                            
    def reset(self):
        self.conversation_log = []
        self.contract = None
        self.agent1.reset()
        self.agent2.reset()

    def setup_custom_logger(self):
        """
        Set up a custom logger with the given name and log file.
        """
        log_file = f'{self.save_dir}/negotiation.ansi'

        formatter = logging.Formatter(fmt='%(message)s')
        handler = logging.FileHandler(log_file, mode='w')  # Change to 'a' if you want to append
        handler.setFormatter(formatter)

        logger = logging.getLogger('negotiation')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        
        def log_and_print(message, print_flag=True):
            logger.info(message)
            if print_flag: print(message)
        return log_and_print
    
    def summarize(self):
        # Prepare a prompt for the summarization
        summary_prompt = "Summarize the following negotiation: \n\n"
        for name, thought, message in self.conversation_log:
            summary_prompt += f"{name} (Thought): {thought}\n"
            summary_prompt += f"{name} (Message): {message}\n\n"

        # Generate a summary using the agent 
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": summary_prompt}],
        )
        summary = response['choices'][0]['message']['content'].strip()
        return summary

    def _display_message(self, log, print_flag=True):
        # Define the color codes
        class Color:
            RED = '\033[91m'
            PINK = '\033[95m'
            BLUE = '\033[94m'
            LIGHT_BLUE = '\033[96m'
            LIGHT_GREEN = '\033[92m'
            GREEN = '\033[92m'
            DARK_GREEN = '\033[32m'
            RESET = '\033[0m'
        
        name, thought, message = log

        # agent1 is blue and agent2 is green
        if name == self.agent1.name:
            self.logger(f"{Color.LIGHT_BLUE}{name} (Thought): {thought}{Color.RESET}",  print_flag=print_flag)
            self.logger(f"{Color.BLUE}{name} (Message): {message}{Color.RESET}\n",  print_flag=print_flag)
        else:
            self.logger(f"{Color.LIGHT_GREEN}{name} (Thought): {thought}{Color.RESET}", print_flag=print_flag)
            self.logger(f"{Color.DARK_GREEN}{name} (Message): {message}{Color.RESET}\n", print_flag=print_flag)

    def simulate(self):
        if len(self.conversation_log) > 0:
            raise Exception("Conversation has already been simulated. Use display() to see the conversation. Or use reset() to start a new conversation.")

        accept = False
        for turn in range(self.max_turns):
            if turn == 0:
                thought, message = self.agent1.generate_message()
                self.conversation_log.append((self.agent1.name, thought, message))
            elif turn % 2 == 0:
                thought, message = self.agent1.generate_message(self.conversation_log[-1][2])
                self.conversation_log.append((self.agent1.name, thought, message))
            else:
                thought, message = self.agent2.generate_message(self.conversation_log[-1][2])
                self.conversation_log.append((self.agent2.name, thought, message))

            # Live display of conversation based on the flag
            self._display_message(self.conversation_log[-1])
            
            # if a player accepts the contract, end the conversation
            if '[accept]' in message:
                accept = True
                break

        # Extract the contract from the conversation log
        if accept:
            try:
                self.contract = self.conversation_log[-2][2].split('[contract]')[1].split('[contract end]')[0].strip()
            except IndexError:
                raise Exception("Negotation failure. Contract accepted but no contract was found. Please try again.")
            self.logger(f"Contract:\n{self.contract}\n")
        else:
            raise Exception("Negotation failure. No contract was found. Please try again.")

        # Summarize the conversation
        summary = self.summarize()
        self.logger(f"\033[90mNegotiation Summary:\n{summary}\n\033[0m", print_flag=False)

    def get_contract(self):
        if self.contract is None:
            raise Exception("Conversation has not been simulated. Use simulate() to simulate the conversation.")
        return self.contract