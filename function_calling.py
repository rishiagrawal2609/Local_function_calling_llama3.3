import pytz
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
import datetime
from langchain_core.tools import tool
import sys
message = []

@tool(parse_docstring=True)
def get_current_time(timezone:str) -> str:
    """
    Returns the current time in the given timezone. call this whenever you need the current time in the perticular timezone. For example when the customers ask "What is the current time in New Yotk?"

    Args:
        timezone (str): IANA timezone name (e.g., 'America/New_York')

    Returns:
        str: current time in the given timezone
    """
    try:
        current_time = datetime.datetime.now(pytz.timezone(timezone))
        return current_time.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        return f"Incorrect timezone {str(e)}"
    
tools_list = {
    "get_current_time": get_current_time
}

prompt = st.text_input("Enter your message here:")

if prompt:
    llm = ChatOllama(model="llama3.3:latest")
    llm_tools = llm.bind_tools(list(tools_list.values()))
    message.append(HumanMessage(prompt))
    response = llm_tools.invoke(message)
    message.append(response.content)

    if not response.tool_calls:
        with st.container(height=500,border=True):
            st.write(response.content)
            sys.exit()
    
    for tool_call in response.tool_calls:
        selected_tool = tools_list[tool_call["name"].lower()]
        target_response = selected_tool.invoke(tool_call["args"])
        message.append(ToolMessage(target_response, tool_call_id=tool_call["id"]))

    final_response = llm_tools.stream(message)
    with st.container(height=500,border=True):
        st.write_stream(final_response)


