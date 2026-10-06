import uuid

import pytest
from langchain_core.messages import AIMessage

import main
from main import (
    ClimaInput,
    ErrorDelAgente,
    HumanMessage as Humano,
    app,
    mensaje_validacion,
)


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
        {"messages": [Humano(content="¿Cuál es el clima en Montevideo?")]},
        config=_hilo(),
    )
    tipos = [mensaje.type for mensaje in salida["messages"]]
    assert tipos == ["human", "ai", "tool", "ai"]
    assert _tool_calls(salida)[0]["name"] == "get_weather"
    assert _tool_calls(salida)[0]["args"]["location"] == "Montevideo"
    assert salida["messages"][2].content == main.CLIMA["montevideo"]
    assert salida["messages"][-1].tool_calls == []
    assert salida["messages"][-1].content.startswith("En Montevideo:")


@pytest.mark.asyncio
async def test_mismo_thread_id_recuerda_la_ubicacion():
    config = _hilo()
    await app.ainvoke(
        {"messages": [Humano(content="¿Cuál es el clima en Montevideo?")]},
        config=config,
    )
    salida = await app.ainvoke(
        {"messages": [Humano(content="¿Y mañana en el mismo lugar?")]},
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
    primero = _hilo()
    await app.ainvoke(
        {"messages": [Humano(content="¿Cuál es el clima en Montevideo?")]},
        config=primero,
    )
    otro = _hilo()
    salida = await app.ainvoke(
        {"messages": [Humano(content="¿Y mañana en el mismo lugar?")]},
        config=otro,
    )
    assert salida["messages"][-1].content == "Decime la ciudad para consultar el clima."
    assert _tool_calls(salida) == []
    assert "Montevideo" not in salida["messages"][-1].content


@pytest.mark.asyncio
async def test_cambia_de_ciudad_en_el_mismo_hilo():
    config = _hilo()
    await app.ainvoke(
        {"messages": [Humano(content="¿Cuál es el clima en Montevideo?")]},
        config=config,
    )
    salida = await app.ainvoke(
        {"messages": [Humano(content="¿Cuál es el clima en Madrid?")]},
        config=config,
    )
    assert _tool_calls(salida)[-1]["args"]["location"] == "Madrid"
    assert salida["messages"][-1].content.startswith("En Madrid:")


@pytest.mark.asyncio
async def test_pregunta_que_no_es_clima_no_llama_la_herramienta():
    salida = await app.ainvoke(
        {"messages": [Humano(content="¿Cuánto es 2+2?")]},
        config=_hilo(),
    )
    assert _tool_calls(salida) == []
    assert salida["messages"][-1].content == "Eso no es una consulta de clima."


@pytest.mark.asyncio
async def test_clima_sin_ciudad_no_ejecuta_get_weather():
    salida = await app.ainvoke(
        {"messages": [Humano(content="¿Cuál es el clima?")]},
        config=_hilo(),
    )
    assert _tool_calls(salida) == []
    assert "Decime la ciudad" in salida["messages"][-1].content


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
    primero = Humano(content="uno")
    segundo = Humano(content="dos")
    acumulado = main.add_messages([primero], [segundo])
    assert [mensaje.content for mensaje in acumulado] == ["uno", "dos"]


def test_clima_input_rechaza_location_en_blanco():
    with pytest.raises(main.ValidationError) as exc:
        ClimaInput(location="   ")
    assert "location no puede estar vacía" in mensaje_validacion(exc.value)


def test_clima_input_rechaza_entero_y_campo_extra():
    with pytest.raises(main.ValidationError):
        ClimaInput(location=18)
    with pytest.raises(main.ValidationError) as extra:
        ClimaInput(location="Montevideo", unidad="C")
    assert "Extra inputs are not permitted" in mensaje_validacion(extra.value)


def test_get_weather_devuelve_el_parte_de_la_ciudad():
    assert main.get_weather.invoke({"location": "Montevideo"}) == main.CLIMA["montevideo"]
    assert main.get_weather.invoke({"location": "  buenos   aires "}) == "Soleado, 22°C"
    assert main.get_weather.invoke({"location": "Lima"}) == "Soleado, 22°C"


def test_get_weather_func_valida_espacios():
    texto = main.get_weather.func("   ")
    assert texto.startswith("Error de validación:")
    assert "location no puede estar vacía" in texto


@pytest.mark.asyncio
async def test_location_en_blanco_vuelve_como_observacion(monkeypatch):
    class _LocationVacia:
        async def ainvoke(self, messages):
            if isinstance(messages[-1], main.ToolMessage):
                return AIMessage(content="Sin ciudad no consulto el clima.")
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"location": "   "},
                        "id": "call_vacia",
                        "type": "tool_call",
                    }
                ],
            )

    monkeypatch.setattr(main, "model", _LocationVacia())
    salida = await app.ainvoke(
        {"messages": [Humano(content="¿Cuál es el clima?")]},
        config=_hilo(),
    )
    observacion = next(mensaje.content for mensaje in salida["messages"] if mensaje.type == "tool")
    assert observacion.startswith("Error de validación: location:")
    assert "location no puede estar vacía" in observacion
    assert salida["messages"][-1].content == "Sin ciudad no consulto el clima."


class _QueFalla:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.llamadas = 0

    async def ainvoke(self, messages):
        self.llamadas += 1
        raise self.exc


class _QueTrunca:
    def __init__(self) -> None:
        self.llamadas = 0

    async def ainvoke(self, messages):
        self.llamadas += 1
        return AIMessage(
            content="parte cortado",
            response_metadata={"finish_reason": "length"},
        )


@pytest.mark.asyncio
async def test_timeout_no_borra_el_checkpoint(monkeypatch):
    config = _hilo()
    await app.ainvoke(
        {"messages": [Humano(content="¿Cuál es el clima en Montevideo?")]},
        config=config,
    )
    fallo = _QueFalla(TimeoutError("Request timed out talking to the model"))
    monkeypatch.setattr(main, "model", fallo)
    with pytest.raises(ErrorDelAgente) as exc:
        await app.ainvoke(
            {"messages": [Humano(content="¿Y mañana en el mismo lugar?")]},
            config=config,
        )
    assert str(exc.value).startswith("red/timeout:")
    assert "ainvoke(None)" in str(exc.value)
    assert fallo.llamadas == 1
    estado = await app.aget_state(config)
    assert estado.next == ("agent",)
    textos = [mensaje.content for mensaje in estado.values["messages"] if isinstance(mensaje.content, str)]
    assert any("Montevideo" in texto for texto in textos)
    monkeypatch.setattr(main, "model", main.ModeloClimaLocal())
    reanudado = await app.ainvoke(None, config=config)
    assert reanudado["messages"][-1].content.startswith("Para mañana en Montevideo")
    humanos = [mensaje.content for mensaje in reanudado["messages"] if mensaje.type == "human"]
    assert humanos.count("¿Y mañana en el mismo lugar?") == 1


@pytest.mark.asyncio
async def test_401_no_reintenta(monkeypatch):
    fallo = _QueFalla(main.FalloSimulado(401, "Unauthorized: invalid API key"))
    monkeypatch.setattr(main, "model", fallo)
    with pytest.raises(ErrorDelAgente) as exc:
        await app.ainvoke(
            {"messages": [Humano(content="¿Cuál es el clima en Madrid?")]},
            config=_hilo(),
        )
    assert str(exc.value).startswith("401/key:")
    assert "No reintentes" in str(exc.value)
    assert fallo.llamadas == 1


@pytest.mark.asyncio
async def test_429_un_solo_intento(monkeypatch):
    fallo = _QueFalla(main.FalloSimulado(429, "rate limit / quota exceeded"))
    monkeypatch.setattr(main, "model", fallo)
    with pytest.raises(ErrorDelAgente) as exc:
        await app.ainvoke(
            {"messages": [Humano(content="¿Cuál es el clima en Madrid?")]},
            config=_hilo(),
        )
    assert str(exc.value).startswith("429/cuota:")
    assert fallo.llamadas == 1


@pytest.mark.asyncio
async def test_error_no_transitorio_un_solo_intento(monkeypatch):
    fallo = _QueFalla(RuntimeError("boom"))
    monkeypatch.setattr(main, "model", fallo)
    with pytest.raises(ErrorDelAgente) as exc:
        await app.ainvoke(
            {"messages": [Humano(content="¿Cuál es el clima en Madrid?")]},
            config=_hilo(),
        )
    assert str(exc.value).startswith("Error del agente: boom")
    assert "ainvoke(None)" in str(exc.value)
    assert fallo.llamadas == 1


@pytest.mark.asyncio
async def test_salida_truncada_no_entra_al_historial(monkeypatch):
    config = _hilo()
    trunca = _QueTrunca()
    monkeypatch.setattr(main, "model", trunca)
    with pytest.raises(ErrorDelAgente) as exc:
        await app.ainvoke(
            {"messages": [Humano(content="¿Cuál es el clima en Madrid?")]},
            config=config,
        )
    assert str(exc.value).startswith("Salida truncada:")
    assert trunca.llamadas == 1
    estado = await app.aget_state(config)
    contenidos = [getattr(m, "content", "") for m in estado.values["messages"]]
    assert "parte cortado" not in contenidos
