import os
from api_key import apikey
import subprocess

os.environ["OPENAI_API_KEY"] = apikey

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import ShellTool
from langchain.tools import Tool
from langchain.chains import LLMChain

# Init the LLM which powers the agent.
llm = ChatOpenAI(temperature=0,model='gpt-3.5-turbo')
    

# Init the terminal tool.
terminal = Tool(
        name = "terminal",
        func=ShellTool().run,
        description="useful to run shell commands on this Linux machine."
)

# Init the toolkit needed.
tools = [terminal]

# Init the agent.
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_error =True 
    )

# Creating a prompt_template.
template = """
from the given paragraph extract the needed information and make a prompt needed to make a Dockerfile according to the paragraph.
Do not miss out on any information.

paragraph:
{details}

DO NOT GIVE THE DOCKERFILE CONTENTS JUST GIVE THE PROMPT.
"""

# details = input("Please provide the details of the dockerfile: ")
details = '''
I want a Dockerfile for a Node.js application.
It should use Node.js version 14 and require the following dependencies: Express, MongoDB driver, and dotenv.
Please use the official Node.js base image.
'''

prompt_template = PromptTemplate.from_template(template=template)


chain = LLMChain(
    llm=llm,
    prompt=prompt_template
)

# Creating a prompt from the user's details using a simple LLMChain.
prompt = chain.run(details)
template2 = """\nREMEMBER: Only make changes to the Dockerfile through echo commands.
Also try to build the docker image and also try to resolve any errors you encounter especially during dependency installation or application building.
If the same error persists for 3 or more times exit by returning the error code and the reason.
"""

# Creating the input prompt for the agent.
FINAL_PROMPT = prompt+template2

#print(prompt)
#print(agent.agent.llm_chain.prompt.template)

agent.run(FINAL_PROMPT)