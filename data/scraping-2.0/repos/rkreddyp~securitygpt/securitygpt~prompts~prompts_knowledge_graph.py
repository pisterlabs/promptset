from typing import List, Optional, Type, Union
from enum import Enum, auto
from pydantic.dataclasses import dataclass
from pydantic import BaseModel, Field, create_model
import openai, boto3, requests, json

from securitygpt.schema.schema_base_prompt import SystemMessage, UserMessage, PromptContext, ChatCompletionMessage, MessageRole, Message




class PromptCVERole(BaseModel):
    role_string: str = Field(default="na")
    role_string = """ 
    # Your Role
    
     """
    @property
    def return_string(self):
        return self.role_string
   

class PromptKnowledgeGraph(BaseModel):
    prompt_string: str = Field(default="na")
    prompt_string = """ 
        Your task is make the knowledge graph from an article for a given objective.
        objective : {objective}
        you must follow all reqirements that are listed below
        ## graph requirements
        - in the edges list, count the number of target ids for each source ids, and the
        number must be list_target_ids for each node
        - when you make the edges, ensure that there must not be more than 10 target node 
            ids connected to any given source node id in the graph. the number in list_target_ids
            for any given source node id must therefore be at most 10
        - for edge colors, you must make it visually informative.  you must use red color for 
            important bad items , green color for good items, be creative for other colors
        - there graph should be uni-directional, no bi-directional edges.  for example,
            if node 1 is connected to node 2, then node 2 should not be connected to node 1
        - the number of target nodes that are attached to source node of id 1 should be at most 5
        - group all related targets belonging to a theme to one soruce id
        - all nodes in the graph should be connected to atleast one other node
        - all labels should be text and strings, not integers
        - num_edges for all nodes should be 2 or more
        - all nodes should be connected to node with source id 1 , directly or indirectly
        - focus on specific information such time periods, dates, numbers , percentages, make them as edges such as "on" or "during"
        - if time periods are mentioned, use them as edges and put the nodes and IDs in chronological order
        - ensure to be as exhaustive as you can, include all the details from the article to acheive the objective
        
        # example of a good graph output
        ```
            {{'nodes': [{{'id': 1,
            'label': 'Attack Details (SOFARPC)',
            'color': 'blue',
            'num_targets': 4,
            'num_sources': 0,
            'list_target_ids': [3,4,8,18],
            'num_edges': 4}},
            {{'id': 2,
            'label': 'Java RPC framework',
            'color': 'blue',
            'num_targets': 1,
            'num_sources': 1,
            'list_target_ids': [1],
            'num_edges': 2}},
            {{'id': 3,
            'label': 'Versions prior to 5.11.0',
            'color': 'blue',
            'num_targets': 1,
            'num_sources': 1,
            'list_target_ids': [1],
            'num_edges': 2}},
            {{'id': 4,
            'label': 'Remote command execution',
            'color': 'red',
            'num_targets': 1,
            'num_sources': 2,
            'list_target_ids': [1,5],
            'num_edges': 3}},
            {{'id': 5,
            'label': 'Payload',
            'color': 'red',
            'num_targets': 2,
            'num_sources': 1,
            'list_target_ids': [4,6,7],
            'num_edges': 3}},
            {{'id': 6,
            'label': 'JNDI injection',
            'color': 'red',
            'num_targets': 1,
            'num_sources': 1,
            'list_target_ids': [5],
            'num_edges': 2}},
            {{'id': 7,
            'label': 'System command execution',
            'color': 'red',
            'num_targets': 1,
            'num_sources': 1,
            'list_target_ids': [5],
            'num_edges': 2}},
            {{'id': 9,
            'label': 'Version 5.11.0',
            'color': 'green',
            'num_targets': 1,
            'num_sources': 1,
            'list_target_ids': [8],
            'num_edges': 2}},
            {{'id': 18,
            'label': 'Fix',
            'color': 'green',
            'num_targets': 1,
            'num_sources': 1,
            'list_target_ids': [1],
            'num_edges': 2}},
            {{'id': 8,
            'label': 'Workarounds and Fixe Details',
            'color': 'green',
            'num_targets': 1,
            'num_sources': 1,
            'list_target_ids': [9],
            'num_edges': 2}},
            {{'id': 11,
            'label': '-Drpc_serialize_blacklist_override=javax.sound.sampled.AudioFileFormat',
            'color': 'green',
            'num_targets': 1,
            'num_sources': 1,
            'list_target_ids': [18],
            'num_edges': 2}},
            ],
        'edges': [{{'source': 1, 'target': 2, 'label': 'is a', 'color': 'blue'}},
            {{'source': 1, 'target': 3, 'label': 'Versions prior to', 'color': 'blue'}},
            {{'source': 1, 'target': 4, 'label': 'vulnerable to', 'color': 'red'}},
            {{'source': 4, 'target': 5, 'label': 'can achieve', 'color': 'blue'}},
            {{'source': 5, 'target': 6, 'label': 'or', 'color': 'red'}},
            {{'source': 5, 'target': 7, 'label': 'or', 'color': 'red'}},
            {{'source': 1, 'target': 8, 'label': 'work around', 'color': 'red'}},
            {{'source': 8, 'target': 9, 'label': 'or', 'color': 'green'}},
            {{'source': 1, 'target': 18, 'label': 'fixed by', 'color': 'red'}},
            {{'source': 18, 'target': 11, 'label': 'or', 'color': 'green'}},
            
            ]}}
        ```
        
        think step by step, 
        before you give the graph to the user =
        - are the list_target_ids filled up for all nodes ?
        - is the number of list_target_ids (num_targets) more than 8 ?
        then re-do the graph

        the attack article is as follows :  {article}
    """

    @property
    def return_string(self):
        return self.prompt_string

# may be goal
class PromptCVEOutput(BaseModel):
    output_string: str = Field(default="na")
    output_string = """ 
    # Your response 
    Your response should be a verdict on the email whether its malicious email or not,
     with along with reasons.  Each of the reasons should be a probability that its a CVE
     because of that reason.  The response should be in json format. 

    """
    @property
    def return_string(self):
        return self.output_string
        
class PromptCVEExample(BaseModel):
    example_string: str = Field(default="na")
    example_string = """ 
    
    # Example response:
    ```
        {{
            "verdict": "CVEing",
            "CVE_characterestic": {{
                "urgency": 8,
                "lack_of_detail": 8,
                "attachments": 8,
                "generic_salutation": 8,
                "unusual_requests": 8,
                "spelling_and_grammar": 8
            }},
            "over_all_confidence": 8,
            "attack_technique": "spear CVEing"

        }}
        ```
    """
    @property
    def return_string(self):
        return self.example_string
    
class PromptCVEInputString(BaseModel):
    input_string: str = Field(default="na")
    input_string = """ 
    # Email Content
    {email_text}
    """
    @property
    def return_string(self):
        return self.input_string
    
class PrompCVETips(BaseModel):
    tips_string: str = Field(default="na")
    tips_string = """ 
    # Tips  
    Look at the email for the following  attack techniques for business email compromise - - Exploiting Trusted Relationships
    - To urge victims to take quick action on email requests, attackers make a concerted effort to exploit an existing trusted relationship. Exploitation can take many forms, such as a vendor requesting invoice payments, an executive requesting iTunes gift cards, or an [employee sharing new payroll direct deposit details.
    - Replicating Common Workflows
        - An organization and its employees execute an endless number of business workflows each day, many of which rely on automation, and many of which are conducted over email. The more times employees are exposed to these workflows, the quicker they execute tasks from muscle memory. BEC attacks [try to replicate these day-to-day workflows]to get victims to act before they think.
    - Compromised workflows include:
        - Emails requesting a password reset
        - Emails pretending to share files and spreadsheets
        - Emails from commonly used apps asking users to grant them access
    - Suspicious Attachments
        - Suspicious attachments in email attacks are often associated with malware. However, attachments used in BEC attacks forego malware in exchange for fake invoices and other social engineering tactics that add to the conversation’s legitimacy. These attachments are lures designed to ensnare targets further.
    - Socially Engineered Content and Subject Lines
        - BEC emails often rely on subject lines that convey urgency or familiarity and aim to induce quick action.
        - Common terms used in subject lines include:
            - Request
            - Overdue
            - Hello FirstName
            - Payments
            - Immediate Action
        - Email content often follows along the same vein of trickery, with manipulative language that pulls strings to make specific, seemingly innocent requests. Instead of using phishing links, BEC attackers use language as the payload.
    - Leveraging Free Software
        - Attackers make use of freely available software to lend BEC scams an air of legitimacy and help emails sneak past security technologies that block known bad links and domains.
        - For example, attackers use SendGrid to create spoofed email addresses and Google Sites to stand up phishing pages.
    - Spoofing Trusted Domains      

    The following are the categories of business email compromise - - CEO Fraud
    - Attackers impersonate the CEO or executive of a company. As the CEO, they request that an employee within the accounting or finance department transfer funds to an attacker-controlled account.
    - Lawyer Impersonation
        - Attackers pose as a lawyer or legal representative, often over the phone or email. These attacks’ common targets are lower-level employees who may not have the knowledge or experience to question the validity of an urgent legal request.
    - Data Theft
        - Data theft attacks typically target HR personnel to obtain personal information about a company’s CEO or other high-ranking executives. The attackers can then use the data in future attacks like CEO fraud.
    - Email Account Compromise
        - In an [email account compromise]attack, an employee’s email account is hacked and used to request payments from vendors. The money is then sent to attacker-controlled bank accounts.
    - Vendor Email Compromise
        - Companies with foreign suppliers are common targets of [vendor email compromise] Attackers pose as suppliers, request payment for a fake invoice, then transfer the money to a fraudulent account.



    """
    @property
    def return_string(self):
        return self.tips_string

class PromptPhishReflect(BaseModel):
    reflect_string: str = Field(default="na")
    reflect_string = """ 
      
      - Think step by step. Reflect to see if you are producing the correct and complete json format.  if not, then you need to fix your repsponse. 
    
    """
    @property
    def return_string(self):
        return self.reflect_string