import asyncio
import os
import sys
import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

CLIMA = {
    "montevideo": "Nublado, 18°C, viento del sur",
    "buenos aires": "Soleado, 22°C",
    "madrid": "Parcialmente nublado, 16°C",
}


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class ClimaInput(BaseModel):
    """Entrada de get_weather. Tipos estrictos, sin campos extra."""

    model_config = ConfigDict(extra="forbid", strict=True)

    location: str = Field(..., min_length=1, max_length=80, description="Ciudad.")

    @field_validator("location")
    @classmethod
    def location_no_vacia(cls, valor: str) -> str:
        limpio = " ".join(valor.strip().split())
        if not limpio:
            raise ValueError("location no puede estar vacía.")
        return limpio


def mensaje_validacion(exc: ValidationError) -> str:
    err = exc.errors()[0]
    lugar = ".".join(str(parte) for parte in err["loc"]) or "location"
    return (
        f"Error de validación: {lugar}: {err['msg']} "
        "Reformulá location: tiene que ser el nombre de la ciudad."
    )


@tool(args_schema=ClimaInput)
def get_weather(location: str) -> str:
    """Consulta el clima actual.

    Args:
        location: ciudad, por ejemplo Montevideo.

    Usala si preguntan el clima, la temperatura o el pronóstico de una ciudad.
    No la uses para cálculos, precios ni charla general.
    """

    try:
        datos = ClimaInput(location=location)
    except ValidationError as exc:
        return mensaje_validacion(exc)
    lugar = datos.location.lower()
    if lugar in CLIMA:
        return CLIMA[lugar]
    return "Soleado, 22°C"


get_weather.handle_validation_error = mensaje_validacion

tools = [get_weather]
tool_node = ToolNode(tools)
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY", "sk-test-no-usar"),
).bind_tools(tools)


async def call_model(state: AgentState):
    """Pide al LLM el siguiente paso del ciclo: otra herramienta, o la respuesta final.

    Args:
        state: historial en messages. add_messages concatena; no pisa el hilo.

    No ejecuta get_weather. Si hay tool_calls, tools_condition manda el turno
    al nodo tools y la observación vuelve a este nodo.
    """

    try:
        response = await model.ainvoke(state["messages"])
    except Exception as exc:
        raise ErrorDelAgente(mensaje_de_fallo(exc)) from exc
    meta = getattr(response, "response_metadata", None) or {}
    if meta.get("finish_reason") == "length":
        raise ErrorDelAgente(
            "Salida truncada: el modelo cortó la respuesta. "
            "Reformulá la consulta y reanudá con ainvoke(None) y el mismo thread_id."
        )
    return {"messages": [response]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", "__end__": END},
)
workflow.add_edge("tools", "agent")
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


class ErrorDelAgente(Exception):
    """El nodo agent cortó el turno. El checkpoint anterior de ese thread_id sigue."""


class FalloSimulado(Exception):
    """Error del mock (status HTTP). No sale a ninguna API real."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        super().__init__(detail)


def mensaje_de_fallo(exc: BaseException) -> str:
    """Texto del fallo. No reintenta: el hilo se reanuda con el mismo thread_id."""

    detalle = str(exc).strip() or exc.__class__.__name__
    texto = detalle.lower()
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if "finish_reason=length" in texto or "salida truncada" in texto:
        return (
            "Salida truncada: el modelo cortó la respuesta. "
            f"Detalle: {detalle} Reformulá la consulta y reanudá con ainvoke(None) "
            "y el mismo thread_id."
        )
    if status == 401 or "invalid api key" in texto or "unauthorized" in texto:
        return (
            "401/key: clave rechazada o ausente. "
            f"Detalle: {detalle} No reintentes: corregí la clave y reanudá con "
            "ainvoke(None) y el mismo thread_id."
        )
    if status == 429 or "rate limit" in texto or "quota" in texto:
        return (
            "429/cuota: el modelo rechazó el turno por cuota o rate limit. "
            f"Detalle: {detalle} Reintentá más tarde con ainvoke(None) y el mismo thread_id."
        )
    if (
        isinstance(exc, (TimeoutError, ConnectionError))
        or "timed out" in texto
        or "timeout" in texto
    ):
        return (
            "red/timeout: no se pudo hablar con el modelo. "
            f"Detalle: {detalle} Reanudá con ainvoke(None) y el mismo thread_id. "
            "No reenvíes el mensaje."
        )
    return (
        f"Error del agente: {detalle} "
        "Reanudá con ainvoke(None) y el mismo thread_id si el fallo fue transitorio."
    )


class ModeloClimaLocal:
    """Misma firma ainvoke que el modelo con bind_tools. Lee el historial del estado.

    No guarda la ciudad en atributos: si el checkpoint no reinyecta los mensajes,
    el segundo turno no puede recuperar la ubicación. No llama a la red.
    """

    async def ainvoke(self, messages):
        ultimo = messages[-1]
        if isinstance(ultimo, ToolMessage):
            return _cerrar_clima(messages, str(ultimo.content))
        if _pide_clima(messages):
            ciudad = _ciudad_en_historial(messages)
            if ciudad is None:
                return AIMessage(content="Decime la ciudad para consultar el clima.")
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"location": ciudad},
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="Eso no es una consulta de clima.")


def _ultimo_humano(messages) -> str:
    for mensaje in reversed(messages):
        if getattr(mensaje, "type", "") == "human" and isinstance(mensaje.content, str):
            return mensaje.content
    return ""


def _pide_clima(messages) -> bool:
    ultimo = _ultimo_humano(messages).lower()
    consulta = any(
        palabra in ultimo
        for palabra in ("clima", "temperatura", "lluvia", "pronóstico", "pronostico")
    )
    seguimiento = any(
        palabra in ultimo
        for palabra in ("mañana", "manana", "mismo lugar", "ahí", "ahi", "allí", "alli")
    )
    return consulta or seguimiento


def _ciudad_en_historial(messages) -> str | None:
    nombres = (
        ("buenos aires", "Buenos Aires"),
        ("montevideo", "Montevideo"),
        ("madrid", "Madrid"),
    )
    hallada = None
    for mensaje in messages:
        texto = getattr(mensaje, "content", "")
        if not isinstance(texto, str):
            continue
        bajo = texto.lower()
        mejor_pos = -1
        mejor = None
        for clave, nombre in nombres:
            pos = bajo.rfind(clave)
            if pos > mejor_pos:
                mejor_pos = pos
                mejor = nombre
        if mejor is not None:
            hallada = mejor
    return hallada


def _cerrar_clima(messages, observacion: str) -> AIMessage:
    ciudad = _ciudad_en_historial(messages) or "ese lugar"
    humano = _ultimo_humano(messages).lower()
    if "mañana" in humano or "manana" in humano:
        return AIMessage(
            content=(
                f"Para mañana en {ciudad} el parte disponible es el de hoy: {observacion}"
            )
        )
    return AIMessage(content=f"En {ciudad}: {observacion}")


def usar_modelo_local() -> None:
    """La demo corre el grafo sin red. CODER_USAR_OPENAI=1 deja ChatOpenAI.bind_tools."""

    global model
    if os.getenv("CODER_USAR_OPENAI") == "1":
        return
    model = ModeloClimaLocal()


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _imprimir_turno(salida: dict) -> None:
    for mensaje in salida["messages"]:
        tipo = getattr(mensaje, "type", "")
        if tipo == "human":
            print(f"humano: {mensaje.content}")
        elif tipo == "ai" and getattr(mensaje, "tool_calls", None):
            for llamada in mensaje.tool_calls:
                args = llamada.get("args", {})
                print(f"tool_calls: {llamada.get('name')} location={args.get('location')}")
        elif tipo == "tool":
            print(f"observacion: {mensaje.content}")
        elif tipo == "ai":
            print(f"respuesta: {mensaje.content}")


async def main() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")

    usar_modelo_local()
    thread_config = {"configurable": {"thread_id": "dev_test_01"}}

    print("--- Paso 1 ---")
    paso_1 = await app.ainvoke(
        {"messages": [HumanMessage(content="¿Cuál es el clima en Montevideo?")]},
        config=thread_config,
    )
    _imprimir_turno(paso_1)

    print("--- Paso 2 ---")
    paso_2 = await app.ainvoke(
        {"messages": [HumanMessage(content="¿Y mañana en el mismo lugar?")]},
        config=thread_config,
    )
    _imprimir_turno(paso_2)

    estado = await app.aget_state(thread_config)
    humanos = [
        mensaje.content
        for mensaje in estado.values["messages"]
        if getattr(mensaje, "type", "") == "human"
    ]
    print(f"checkpoint thread_id=dev_test_01 humanos={humanos}")
    if "Montevideo" not in paso_2["messages"][-1].content:
        print("Error controlado: el segundo turno no recordó la ubicación")
        raise SystemExit(1)
    if "Montevideo" not in humanos[0] or "Montevideo" in humanos[1]:
        print(f"Error controlado: el checkpoint no separó los turnos: {humanos}")
        raise SystemExit(1)

    print("--- Otro hilo ---")
    otro = await app.ainvoke(
        {"messages": [HumanMessage(content="¿Y mañana en el mismo lugar?")]},
        config=_config("dev_test_02"),
    )
    print(otro["messages"][-1].content)
    if "Montevideo" in otro["messages"][-1].content:
        print("Error controlado: otro thread_id arrastró la ubicación")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
