# Going to change the underlying agent into Plan and Execute Agent to check the efficency.

import os
from api_key import apikey

from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.tools import ShellTool
from langchain.agents.tools import Tool
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


os.environ["OPENAI_API_KEY"] = apikey

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

planner = load_chat_planner(llm)

executor = load_agent_executor(llm, tools, verbose=True)

agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

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
I need a Dockerfile for my Node.js application. 
It requires Node.js version 12, npm, and the Express framework. 
The base image should be the latest Node.js Alpine image. 
Additionally, I would like to include the following environment variables: PORT=3000, NODE_ENV=production. 
Lastly, please expose port 3000 in the Dockerfile.
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


with get_openai_callback() as cb:
    response = agent.run(FINAL_PROMPT)
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")