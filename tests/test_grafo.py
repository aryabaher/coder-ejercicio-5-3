import uuid
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

import main
from main import app


def _hilo() -> dict:
    return {"configurable": {"thread_id": f"test-{uuid.uuid4().hex}"}}


def _tool_calls(salida: dict) -> list:
    llamadas = []
    for mensaje in salida["messages"]:
        if getattr(mensaje, "type", "") == "ai":
            llamadas.extend(mensaje.tool_calls or [])
    return llamadas


@pytest.mark.asyncio
async def test_ciclo_react_termina_sin_mas_herramientas():
    salida = await app.ainvoke(
        {"messages": [HumanMessage(content="¿Cuál es el clima en Montevideo?")]},
        config=_hilo(),
    )
    tipos = [mensaje.type for mensaje in salida["messages"]]
    assert tipos == ["human", "ai", "tool", "ai"]
    assert _tool_calls(salida)[0]["name"] == "get_weather"
    assert _tool_calls(salida)[0]["args"]["location"] == "Montevideo"
    assert salida["messages"][2].content == "Nublado, 18 C"
    assert salida["messages"][-1].tool_calls == []
    assert salida["messages"][-1].content.startswith("En Montevideo:")


@pytest.mark.asyncio
async def test_mismo_thread_id_recuerda_la_ubicacion():
    config = _hilo()
    await app.ainvoke(
        {"messages": [HumanMessage(content="¿Cuál es el clima en Montevideo?")]},
        config=config,
    )
    salida = await app.ainvoke(
        {"messages": [HumanMessage(content="¿Y mañana en el mismo lugar?")]},
        config=config,
    )
    humanos = [m.content for m in salida["messages"] if m.type == "human"]
    assert "Montevideo" in humanos[0]
    assert "Montevideo" not in humanos[1]
    assert [llamada["args"]["location"] for llamada in _tool_calls(salida)] == [
        "Montevideo",
        "Montevideo",
    ]
    assert salida["messages"][-1].content.startswith("Para mañana en Montevideo")
    estado = await app.aget_state(config)
    assert len(estado.values["messages"]) == len(salida["messages"])


@pytest.mark.asyncio
async def test_otro_thread_id_no_arrastra_la_ciudad():
    await app.ainvoke(
        {"messages": [HumanMessage(content="¿Cuál es el clima en Montevideo?")]},
        config=_hilo(),
    )
    salida = await app.ainvoke(
        {"messages": [HumanMessage(content="¿Y mañana en el mismo lugar?")]},
        config=_hilo(),
    )
    assert salida["messages"][-1].content == "Decime la ciudad para consultar el clima."
    assert _tool_calls(salida) == []


@pytest.mark.asyncio
async def test_cambia_de_ciudad_en_el_mismo_hilo():
    config = _hilo()
    await app.ainvoke(
        {"messages": [HumanMessage(content="¿Cuál es el clima en Montevideo?")]},
        config=config,
    )
    salida = await app.ainvoke(
        {"messages": [HumanMessage(content="¿Cuál es el clima en Madrid?")]},
        config=config,
    )
    assert _tool_calls(salida)[-1]["args"]["location"] == "Madrid"
    assert salida["messages"][-1].content.startswith("En Madrid:")


@pytest.mark.asyncio
async def test_pregunta_que_no_es_clima_no_llama_la_herramienta():
    salida = await app.ainvoke(
        {"messages": [HumanMessage(content="¿Cuánto es 2+2?")]},
        config=_hilo(),
    )
    assert _tool_calls(salida) == []
    assert salida["messages"][-1].content == "Eso no es una consulta de clima."


def test_grafo_tiene_nodos_y_checkpointer():
    grafo = app.get_graph()
    assert {"agent", "tools", "__start__", "__end__"} <= set(grafo.nodes)
    aristas = {(arista.source, arista.target) for arista in grafo.edges}
    assert ("__start__", "agent") in aristas
    assert ("agent", "tools") in aristas
    assert ("tools", "agent") in aristas
    assert ("agent", "__end__") in aristas
    assert isinstance(main.memory, main.MemorySaver)
    assert app.checkpointer is main.memory


def test_add_messages_concatena():
    primero = HumanMessage(content="uno")
    segundo = HumanMessage(content="dos")
    acumulado = main.add_messages([primero], [segundo])
    assert [mensaje.content for mensaje in acumulado] == ["uno", "dos"]


def test_get_weather_devuelve_el_parte_de_la_ciudad():
    assert main.get_weather.invoke({"location": "Montevideo"}) == "Nublado, 18 C"
    assert main.get_weather.invoke({"location": " madrid "}) == "Parcialmente nublado, 16 C"
    assert main.get_weather.invoke({"location": "Lima"}) == "Soleado, 22 C"


def test_el_modelo_del_modulo_es_chatopenai():
    fuente = Path(main.__file__).read_text(encoding="utf-8")
    assert fuente.count("StateGraph(AgentState)") == 1
    assert "ModeloClimaLocal" not in fuente
    assert "usar_modelo_local" not in fuente
    assert "ChatOpenAI" in fuente
    assert ".bind_tools([get_weather])" in fuente
    assert len(fuente.splitlines()) < 65
    assert "sk-" not in fuente
    assert fuente.isascii()
