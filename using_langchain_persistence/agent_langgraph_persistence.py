import os

from langgraph.constants import END
from langgraph.graph import Graph, StateGraph
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

from langchain.tools import tool

from langchain_aws import ChatBedrock
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from llm_based_agents import aws_tokens

os.environ['AWS_ACCESS_KEY'] = aws_tokens.AWS_ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = aws_tokens.AWS_SECRET_ACCESS_KEY
os.environ['AWS_SESSION_TOKEN'] = aws_tokens.AWS_SESSION_TOKEN
os.environ['AWS_REGION'] = "us-east-1"


# LangSmith configuration - set via environment variables
# Do not hardcode API keys in source code
os.environ['LANGSMITH_TRACING'] = os.getenv('LANGSMITH_TRACING', 'true')
os.environ['LANGSMITH_ENDPOINT'] = os.getenv('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY', '')
os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT', 'Shopping-Assistant')


@tool
def get_offers_by_retailer(retailer: str) -> str:
    """Return offers available on a spcific retialer"""
    return ("retailer" + retailer + "has 20% off on all products")


@tool
def get_nearest_store(location: str) -> str:
    """Return the nearest store to the given location."""
    return ("The nearest store is at 123 Main St")


@tool
def get_offers_for_product(product: str) -> str:
    """Return the current offers for the specified product."""
    return ("Offer of" + product + "is 20% off")


@tool
def add_to_shopping_cart(product: str) -> str:
    """Add the specified product to the shopping cart."""
    return ("Added" + product + "to cart")


# Agent state will be a list of messages, where each message can be a SystemMessage, HumanMessage, AIMessage or
# ToolMessage.
# operator.add is used to concatenate the messages, so all the messages will be in the same list,
# maintaining all the context of the conversation.
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# Intialize the persistent storage "CheckPointer", in memory,
# it can be replaced with any other supported storage, like Redis, Postgres.
# also it can be connected to external DB.
# memory = SqliteSaver.from_conn_string(":memory:") this line is not working, replacement is below
sqlite3_conn = sqlite3.connect('checkpoints.sqlite', check_same_thread=False)
sqlite3_memory_checkpoint = SqliteSaver(sqlite3_conn)


class Agent:
    def __init__(self, model, tools, checkpointer, system_prompt=""):
        self.system_prompt = system_prompt
        # initiate graph passing the state
        self.graph = self.build_graph(checkpointer)
        # Create dictionary of tools, each tool name is a key and the value is the tool itself.
        self.tools = {tool.name: tool for tool in tools}
        # Bind the tools to the model, mandatory so the model can return the correct tool to be called.
        self.model = model.bind_tools(tools)

    # Design the graph
    def build_graph(self, checkpointer):
        graph = StateGraph(AgentState)
        # Start point
        graph.set_entry_point("llm")
        # First node to call the LLM, as we are going to pass the system prompt(SystemMessage) and the question(
        # HumanMessage).
        graph.add_node("llm", self.call_llm)
        # Conditional edge to check if there is any action to be executed, if so, go to the action node, otherwise END.
        graph.add_conditional_edges("llm", self.is_action_exist, {True: "action", False: END})
        # Action node to execute the tools, based on the LLM feedback, on which tool to use.
        graph.add_node("action", self.take_action)
        # Edge to go back to the LLM node, to continue the conversation, passing the output of the actions.
        graph.add_edge("action", "llm")
        return graph.compile(checkpointer=checkpointer)

    ## Call the LLM, passing the state which includes all the messages.
    # and each time we concatenate system prompt with the list of messages so LLM knows all the context.
    def call_llm(self, state: AgentState):
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        llm_response = self.model.invoke(messages)
        print("LLM Response: ", llm_response)

        # Return messages, llm_response, Out of the box, LangGraph will concatenate the llm response to the messages
        # list int the AgentState.
        return {"messages": [llm_response]}

    def is_action_exist(self, state: AgentState):
        last_llm_response_message = state["messages"][-1]

        # inside AIMessage, an attribute tool_calls, is a list, as LLM might return list of tools to be called.
        # example of this attribute: tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'Who won
        # the Super Bowl in 2024'}, 'id': 'toolu_bdrk_01UYG3MioqTSThK1YfGojHBA', 'type': 'tool_call'}]

        # here we check if any tool to be called, so method return True to go to the action node.
        return len(last_llm_response_message.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls_in_the_last_message = state["messages"][-1].tool_calls
        actions_results = []
        # Loop through the tools to be called, as LLM might return multiple tools to be called.
        for tool_call in tool_calls_in_the_last_message:
            print(f"Calling tool: {tool_call}")
            if not tool_call["name"] in self.tools:
                raise ValueError(f"Tool {tool_call.tool_name} not found")
            else:
                # Call the tool, passing args
                result = self.tools[tool_call["name"]].invoke(tool_call["args"])
                # create list of ToolMessage each includes the return of the tool/function call
            actions_results.append(
                ToolMessage(tool_call_id=tool_call["id"], name=tool_call["name"], content=str(result)))
        print("Results for the called Actions: ", actions_results)
        # Return the results of the actions, so it's concatenated to the messages list.
        return {"messages": actions_results}


# System Prompt
system_prompt = """You are a shopping assistant that can help users in multiple ways.
you are allowed to ask a follow-up questions to the users and in this case return response.
Be more specific in your answers and reply with only one sentence.
""".strip()

# Initialize Model, it can be replaced with any other supported Model, like ChatOpenAI
bedrock_llm = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_kwargs=dict(temperature=0), aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    aws_session_token=os.environ['AWS_SESSION_TOKEN'],
    region_name=os.environ['AWS_REGION']
)

# initialize the agent
abot = Agent(bedrock_llm,
             [get_nearest_store, get_offers_for_product, get_offers_by_retailer, add_to_shopping_cart],
             sqlite3_memory_checkpoint,
             system_prompt)

# messages = [HumanMessage(content="Add the best offers of butter around me in Lichtenberg to my shopping cart.")]
# # # Call the agent
# abot.graph.invoke({"messages": messages})

# messages = [HumanMessage(content="Please suggest some products to buy around me.")]
# # Call the agent
# abot.graph.invoke({"messages": messages})

# messages = [HumanMessage(content="is there a bread offers around me in lichrtenberg?")]
# # # Call the agent
# abot.graph.invoke({"messages": messages})

messages = [HumanMessage(content="Do Rewe have any banana offers?")]
# thread config to keep track in different threads inside the persistance checkpoint.
# it's really needed for production applications for multiple threads/conversation
thread = {"configurable": {"thread_id": "50"}}

# Streaming messages, AIMessage & ToolMessage
for event in abot.graph.stream({"messages": messages}, thread):
    for value in event.values():
        print(value['messages'])

# another question with the same thread ID, you will notice it already knows the history of the conversation.
messages = [HumanMessage(content="What else do you recommend to buy?")]
thread = {"configurable": {"thread_id": "50"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for value in event.values():
        print(value['messages'])

# ## another question with the same thread ID, you will notice it already knows the history of the conversation.
messages = [HumanMessage(content="in which store I can find those offers?")]
thread = {"configurable": {"thread_id": "50"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for value in event.values():
        print(value['messages'])

# ## another question with the same thread ID, you will notice it already knows the history of the conversation.
messages = [HumanMessage(content="I live in Lichtenberg")]
thread = {"configurable": {"thread_id": "50"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for value in event.values():
        print(value['messages'])
