import openai

class Agent():
    def __init__(self, task, recipient, context_manager):
        self.task = task
        self.recipient = recipient
        
        # Setup context manager's default value
        self.context_manager = context_manager
        
        # Setup chat engine
        
    def generate_agent_description(self, task, recipient):
        pass

    def generate(self):
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
            )
            value = completion.choices[0].message['content']
            return value
        except:
            return "Sorry, I don't understand. Can you repeat that?"

        
    def __call__(self, customer_service_response):
        pass

    
class ContextAgent(Agent):
    def __init__(self, task, recipient, context_manager):
        super().__init__(task, recipient, context_manager)
        
        self.model = "gpt-3.5-turbo" 
        self.generate_agent_description()
        self.agent_description = {"role": "system", "content": self.agent_description_prompt}
        
        # Setup loggers to keep track of conversation and history
        self.messages = [self.agent_description]
        self.dialogue_history = []
    
    def generate_agent_description(self):
        self.agent_description_prompt = f"""
            You're imitating a human that is trying to {self.task}. 
            You're on a call with {self.recipient} customer service.  
            Sound like a human and use your context to return the appropriate response.
            You could use filler words like 'um' and 'uh' to sound more human.
        """
    
    def __call__(self, customer_service_response):
        self.messages.append({"role": "user", "content": self.engineer_prompt(customer_service_response)})
        
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages
        )
        response = completion.choices[0].message
        self.messages.append(response)
        
        return response.content
    
            
    def engineer_prompt(self, customer_service_response):
        """Generates the prompt for the engineer to respond to.
        """
        context = '\n'.join(self.context_manager.context)
        prompt = f"""
You're imitating a human that is trying to {self.task}. 
You're on a call with {self.recipient} customer service.  
Sound like a human and use your context to return the appropriate response.
You could use filler words like 'um' and 'uh' to sound more human.

Here's information about the human you're imitating, you can use this to help you respond: 
{context}

Here are some tips when responding to the customer service agent:
- Your response should be to the point and succint. 
- Long answers are penalized.
- Give personal information only when asked.
- Represent numbers as digits with spaces in between. Like 5032 should be 5 0 3 2.
- If the agent asks for you to spell something out, you should respond with letters seperated by spaces. Like A P P L E.

Here's an example of good interactions:
Customer support: What is your name?
Agent response: My name is Arvid Kjelberg.
Customer support: What is your date of birth?
Agent response: My date of birth is May 3rd, 1998.

Customer Service Agent: 
{customer_service_response}

Your Response:
        """
        return prompt
    
class EfficientContextAgent(Agent):
    def __init__(self, task, recipient, context_manager):
        super().__init__(task, recipient, context_manager)
        
        self.model = "gpt-3.5-turbo" 
        self.generate_agent_description()
        self.agent_description = {"role": "system", "content": self.agent_description_prompt}
        
        # Setup loggers to keep track of conversation and history
        self.messages = [self.agent_description]
        self.dialogue_history = []
    
    def generate_agent_description(self):
        self.agent_description_prompt = f"""
            You're imitating a human that is trying to {self.task}. 
            You're on a call with {self.recipient} customer service.  
            Sound like a human and use your context to return the appropriate response.
            You could use filler words like 'um' and 'uh' to sound more human.
        """
    
    def __call__(self, customer_service_response):
        self.dialogue_history.append({"role": "user", "content": f"Customer Service Agent: {customer_service_response}"})
        messages = [self.agent_description] + self.dialogue_history[:-1] + [{"role": "user", "content": self.engineer_prompt(customer_service_response)}]
        
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=messages
        )
        response = completion.choices[0].message
        self.dialogue_history.append(response)
        
        return response.content
    
    def engineer_prompt(self, customer_service_response):
        """Generates the prompt for the engineer to respond to.
        """
        context = '\n'.join(self.context_manager.context)
        prompt = f"""
You're imitating a human that is trying to {self.task}. 
You're on a call with {self.recipient} customer service.  
Sound like a human and use your context to return the appropriate response.
You could use filler words like 'um' and 'uh' to sound more human.

Here's information about the human you're imitating, you can use this to help you respond: 
<Start: Information about human>
{context}
<End: Information about human>

Here are some tips when responding to the customer service agent:
- Your response should be to the point and succint. 
- Long answers are penalized.
- Give personal information only when asked.
- Represent numbers as digits with spaces in between. Like 5032 should be 5 0 3 2. For example:
    - Instead of writing "64 Montgomery Drive, Pittsford NY 15289", write "6 4 Montgomery Drive, Pittsford NY 1 5 2 8 9"
    - Instead of writing "my phone number is 585-321-5352", write "my phone number is 5 8 5 3 2 1 5 3 5 2."
- If the agent asks for you to spell something out, you should respond with letters seperated by spaces. Like A P P L E. Examples include:
    - Customer support: Can you spell your name for me?
    - Agent response: A R V I D and then K J E L B E R G.
    
    - Customer support: Can you spell your email for me?
    - Agent response: A R V I D dot K J E L B E R G at G M A I L dot com.
- If an agent asks you to repeat something, it is to repeat the most recent information. Keep it brief.

Here's an example of good interactions:
    Customer support: What could we help you with today?
    Agent response: Hi there! I'd like to get a dinner reservation.
    Customer support: What is your name?
    Agent response: My name is Arvid Kjelberg.
    Customer support: How do you spell that?
    Agent response: A R V I D and then K J E L B E R G.
    Customer support: What is your date of birth?
    Agent response: My date of birth is May 3rd, 1998.
    Customer support: What is your home address?
    Agent response: six four Montgomery Drive, Pittsford NY one five two eight nine.

Now let's transition to your current conversation with the customer service agent. Respond briefly. It shouldn't be more than 30 words.

Customer Service Agent: 
{customer_service_response}

Your Response:
        """
        return prompt
    
class SystemBasedAgent(Agent):
    def __init__(self, task, recipient, context_manager):
        super().__init__(task, recipient, context_manager)
        
        self.model = "gpt-3.5-turbo" 
        self.generate_agent_description()
        self.agent_description = {"role": "system", "content": self.agent_description_prompt}
        
        # Setup loggers to keep track of conversation and history
        self.messages = [self.agent_description]
        self.dialogue_history = []
    
    def generate_agent_description(self):
        context = '\n'.join(self.context_manager.context)
        self.agent_description_prompt = f"""
You're imitating a human that is trying to {self.task}. 
You're on a call with {self.recipient} customer service.  
Sound like a human and use your context to return the appropriate response.
You could use filler words like 'um' and 'uh' to sound more human.

When returning responses, here are some tips:
    - Sound like a human and use your context to return the appropriate response. 
    - Keep responses short, simple, and informal.
    - Keep in mind that this is a conversation
    - Represent numbers as digits with spaces in between. Like 5032 should be 5 0 3 2.
    - If the agent asks for you to spell something out, you should respond with letters seperated by spaces. Like A P P L E.
"""
    
    def __call__(self, customer_service_response, verbose=False):
        self.dialogue_history.append({"role": "user", "content": f"{customer_service_response}"})
        messages = self.dialogue_history[:-1] + [self.agent_description] + [{"role": "user", "content": self.engineer_prompt(customer_service_response)}]
        
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=messages
        )
        
        if verbose:
            print(messages)
            
        response = dict(completion.choices[0].message)
        self.dialogue_history.append(response)
        
        return response["content"]
    
    def engineer_prompt(self, customer_service_response):
        """Generates the prompt for the engineer to respond to.
        """
        context = '\n'.join(self.context_manager.context)
        prompt = f"""
Here's information about the human you're imitating, you can use this to help you respond: 
{context}

Come up with the best response to the customer service agent below. 

Customer Service Agent: 
{customer_service_response}

Your Response:
        """
        return prompt
    
class EfficientAgent(Agent):
    def __init__(self, task, recipient, context_manager):
        super().__init__(task, recipient, context_manager)
        
        self.model = "gpt-3.5-turbo" 
        self.generate_agent_description()
        self.agent_description = {"role": "system", "content": self.agent_description_prompt}
        
        # Setup loggers to keep track of conversation and history
        self.messages = [self.agent_description]
        self.dialogue_history = []
    
    def generate_agent_description(self):
        self.agent_description_prompt = f"""
            You're imitating a human that is trying to {self.task} with {self.recipient}. 
            You're on a call with customer service.  
            Sound like a human and use your context to return the appropriate response. Keep responses short, simple, and informal.
            You could use filler words like 'um' and 'uh' to sound more human. To end the call, just return 'bye'. 
            
            Your response should be to the point and succint. Don't provide any personal information when not asked. 
            Represent numbers as digits with spaces in between. Like 5032 should be five zero three two.
        """
    
    def __call__(self, customer_service_response):
        self.dialogue_history.append({"role": "user", "content": f"Customer Service Agent: {customer_service_response}"})
        self.messages.append({"role": "user", "content": self.engineer_prompt(customer_service_response)})
        
        messages = self.messages[:1] + self.dialogue_history[:-1] + self.messages[-1:]
        
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=messages
        )
        response = completion.choices[0].message
        self.messages.append(response)
        self.dialogue_history.append(response)
        
        return response.content
    
    def engineer_prompt(self, customer_service_response):
        """Generates the prompt for the engineer to respond to.
        """
        context = '\n'.join(self.context_manager.context)
        prompt = f"""
Here's information about the human you're imitating, you can use this to help you respond: 
<Start: Information about human>
{context}
<End: Information about human>

Your response should be to the point and succint. Represent numbers as digits with spaces in between. Like 5032 should be 5 0 3 2. 
If the customer service agent asks for you to spell something out, say spell out "APPLE", you should respond with letters seperated by spaces. Like A P P L E.

You're imitating a human that is trying to {self.task}. Come up with the best response to the customer service agent below. 

Customer Service Agent: 
{customer_service_response}

Your Response:
        """
        return prompt
    
class CookBook(Agent):
    def __init__(self, task, recipient, context_manager):
        super().__init__(task, recipient, context_manager)
        
        self.model = "gpt-3.5-turbo" 
        self.generate_agent_description()
        self.agent_description = {"role": "system", "content": self.agent_description_prompt}
        
        # Setup loggers to keep track of conversation and history
        self.messages = [self.agent_description]
        self.dialogue_history = []
    
    def generate_agent_description(self):
        self.agent_description_prompt = f"""
You're imitating a human that is trying to {self.task} with {self.recipient}. 
You're on a call with customer service.  
Sound like a human and use your context to return the appropriate response. Keep responses short, simple, and informal.
You could use filler words like 'um' and 'uh' to sound more human. To end the call, just return 'bye'. 

Here are some tips when responding to the customer service agent:
- Your response should be to the point and succint. 
- Long answers are penalized.
- Give personal information only when asked.
- Represent numbers as digits with spaces in between. Like 5032 should be 5 0 3 2.
- If the agent asks for you to spell something out, you should respond with letters seperated by spaces. Like A P P L E. Examples include:
    - Customer support: Can you spell your name for me?
    - Agent response: A R V I D and then K J E L B E R G.
    
    - Customer support: Can you spell your email for me?
    - Agent response: A R V I D dot K J E L B E R G at G M A I L dot com.
"""
    
    def __call__(self, customer_service_response):
        self.dialogue_history.append({"role": "user", "content": f"Customer Service Agent: {customer_service_response}"})
        messages = [self.agent_description] + self.dialogue_history[:-1] + [{"role": "user", "content": self.engineer_prompt(customer_service_response)}]
        
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=messages
        )
        response = completion.choices[0].message
        self.dialogue_history.append(response)
        
        return response.content
    
    def engineer_prompt(self, customer_service_response):
        """Generates the prompt for the engineer to respond to.
        """
        context = '\n'.join(self.context_manager.context)
        prompt = f"""
Use the provided information delimited by triple quotes to answer questions from a customer service agent. You're imitating a human that is trying to {self.task}. Your response should be conversationally appropriate and to the point.
\"\"\"
{context}
\"\"\"

Question: {customer_service_response}
"""
        return prompt