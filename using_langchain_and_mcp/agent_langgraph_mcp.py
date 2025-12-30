import asyncio
import os

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.constants import END
from langgraph.graph import Graph, StateGraph
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

from langchain.tools import tool

from langchain_aws import ChatBedrock
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from llm_based_agents import aws_tokens

os.environ['AWS_ACCESS_KEY'] = aws_tokens.AWS_ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = aws_tokens.AWS_SECRET_ACCESS_KEY
os.environ['AWS_SESSION_TOKEN'] = aws_tokens.AWS_SESSION_TOKEN
os.environ['AWS_REGION'] = "us-east-1"


# Agent state will be a list of messages, where each message can be a SystemMessage, HumanMessage, AIMessage or
# ToolMessage.
# operator.add is used to concatenate the messages, so all the messages will be in the same list,
# maintaining all the context of the conversation.
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


memory = MemorySaver()

class Agent:
    def __init__(self, model, tools, checkpointer, system_prompt=""):
        self.system_prompt = system_prompt
        # initiate graph passing the state
        self.tools = {tool.name: tool for tool in tools}
        self.graph = self.build_graph(checkpointer, tools)
        self.model = model.bind_tools(tools)

    # Design the graph
    def build_graph(self, checkpointer, tools):
        graph = StateGraph(MessagesState)
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
    def call_llm(self, state: MessagesState):
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        llm_response = self.model.invoke(messages)
        print("LLM Response: ", llm_response)

        # Return messages, llm_response, Out of the box, LangGraph will concatenate the llm response to the messages
        # list int the AgentState.
        return {"messages": [llm_response]}

    def is_action_exist(self, state: MessagesState):
        last_llm_response_message = state["messages"][-1]

        # inside AIMessage, an attribute tool_calls, is a list, as LLM might return list of tools to be called.
        # example of this attribute: tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'Who won
        # the Super Bowl in 2024'}, 'id': 'toolu_bdrk_01UYG3MioqTSThK1YfGojHBA', 'type': 'tool_call'}]

        # here we check if any tool to be called, so method return True to go to the action node.
        return len(last_llm_response_message.tool_calls) > 0

    async def take_action(self, state: MessagesState):
        tool_calls_in_the_last_message = state["messages"][-1].tool_calls
        actions_results = []
        # Loop through the tools to be called, as LLM might return multiple tools to be called.
        for tool_call in tool_calls_in_the_last_message:
            print(f"Calling tool: {tool_call}")
            if not tool_call["name"] in self.tools:
                raise ValueError(f"Tool {tool_call.tool_name} not found")
            else:
                # Call the tool, passing args
                result = await self.tools[tool_call["name"]].ainvoke(tool_call["args"])
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


async def main():
    async with MultiServerMCPClient(
            {
                "content-apis-mcp-server": {
                    "command": "python3",
                    # Make sure to update to the full absolute path to your math_server.py file
                    "args": ["/Users/ramy.hassan/bonial-repos/ai-guild/ai-agents-workshop/ai-agents-workshop/llm_based_agents/local_mcp/local_mcp_server.py"],
                    "transport": "stdio",
                },
                "content-search-apis-mcp-server": {
                    # make sure you start your weather server on port 8000
                    "url": "http://localhost:8000/sse",
                    "transport": "sse",
                }
            }
    ) as client:
        tools = client.get_tools()
        agent = Agent(bedrock_llm, tools, memory, system_prompt)

        messages = [HumanMessage(content="Add the best offers of butter around me in Lichtenberg to my shopping cart.")]

        thread = {"configurable": {"thread_id": "8005"}}
        await agent.graph.ainvoke({"messages": messages}, thread)


        # # Streaming messages, AIMessage & ToolMessage
        # for event in agent.graph.stream({"messages": messages}, thread):
        #     for value in event.values():
        #         print(value['messages'])
        #
        # await agent.graph.ainvoke({"messages": messages}, thread)
        #
        #
        # messages = [HumanMessage(content="Do Rewe have any banana offers?")]
        #
        # await agent.graph.ainvoke({"messages": messages}, thread)
        #
        #
        # messages = [HumanMessage(content="What elese do you recommend to buy?")]
        #
        # await agent.graph.ainvoke({"messages": messages}, thread)

        # async for event in agent.graph.astream_events({"messages": messages}, thread, version="v1"):
        #     kind = event["event"]
        #     if kind == "on_chat_model_stream":
        #         all_text = "".join(part["text"] for part in event["data"]["chunk"].content if part["type"] == "text")
        #         if all_text:
        #             # Empty content in the context of OpenAI means
        #             # that the model is asking for a tool to be invoked.
        #             # So we only print non-empty content
        #             print(all_text, end="")

        messages = [HumanMessage(content="search for a beer offer")]
        await agent.graph.ainvoke({"messages": messages}, thread)

        # async for event in agent.graph.astream_events({"messages": messages}, thread, version="v1"):
        #     kind = event["event"]
        #     if kind == "on_chat_model_stream":
        #         all_text = "".join(part["text"] for part in event["data"]["chunk"].content if part["type"] == "text")
        #         if all_text:
        #             # Empty content in the context of OpenAI means
        #             # that the model is asking for a tool to be invoked.
        #             # So we only print non-empty content
        #             print(all_text, end="")

asyncio.run(main())
