import asyncio
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def get_weather(location: str) -> str:
    """Consulta el clima actual de una ciudad."""
    clima = {"montevideo": "Nublado, 18 C", "madrid": "Parcialmente nublado, 16 C"}
    return clima.get(location.lower().strip(), "Soleado, 22 C")


tool_node = ToolNode([get_weather])
model = ChatOpenAI(model="gpt-4o-mini").bind_tools([get_weather])


async def agent(state: AgentState):
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
# tools -> agent cierra el ciclo ReAct: la observacion vuelve al historial del agente.
workflow.add_edge("tools", "agent")
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


async def main():
    sesion_01 = {"configurable": {"thread_id": "sesion_01"}}
    sesion_02 = {"configurable": {"thread_id": "sesion_02"}}
    pregunta = {"messages": [HumanMessage(content="Cual es el clima en Montevideo?")]}
    seguimiento = {"messages": [HumanMessage(content="Y manana en el mismo lugar?")]}
    await app.ainvoke(pregunta, config=sesion_01)
    await app.ainvoke(seguimiento, config=sesion_01)
    await app.ainvoke(seguimiento, config=sesion_02)
    for nombre, config in (("sesion_01", sesion_01), ("sesion_02", sesion_02)):
        estado = await app.aget_state(config)
        print(f"--- {nombre}: {len(estado.values['messages'])} mensajes en el checkpoint ---")
        for mensaje in estado.values["messages"]:
            mensaje.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
