"""Chequeo offline del ciclo ReAct y del checkpoint (sin API real)."""

from __future__ import annotations

import ast
import asyncio
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

from main import main

ROOT = Path(__file__).resolve().parent
FUENTE = (ROOT / "main.py").read_text(encoding="utf-8")

CADENAS = [
    "class AgentState",
    "messages: Annotated[list[BaseMessage], add_messages]",
    "def get_weather",
    "tool_node = ToolNode(tools)",
    "ChatOpenAI",
    ".bind_tools(tools)",
    "async def call_model",
    'await model.ainvoke(state["messages"])',
    'return {"messages": [response]}',
    "workflow = StateGraph(AgentState)",
    'workflow.add_node("agent", call_model)',
    'workflow.add_node("tools", tool_node)',
    'workflow.add_edge(START, "agent")',
    "tools_condition",
    'workflow.add_edge("tools", "agent")',
    "memory = MemorySaver()",
    "app = workflow.compile(checkpointer=memory)",
    'thread_id": "dev_test_01"',
    "ainvoke",
    "401/key",
    "429/cuota",
    "red/timeout",
    "Salida truncada",
    "No reintentes",
    'if __name__ == "__main__":',
    "asyncio.run(main())",
]


def _configure_stdio() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")


def _ok(nombre: str, condicion: bool, detalle: str = "") -> None:
    estado = "OK" if condicion else "FALLO"
    extra = f" - {detalle}" if detalle else ""
    print(f"[{estado}] {nombre}{extra}")
    if not condicion:
        raise SystemExit(1)


def _chequear_cuerpo(arbol: ast.Module) -> None:
    for nodo in arbol.body:
        if isinstance(nodo, ast.AsyncFunctionDef) and nodo.name == "call_model":
            fin = nodo.end_lineno or nodo.lineno
            cuerpo = "\n".join(FUENTE.splitlines()[nodo.lineno - 1 : fin])
            _ok("call_model está al inicio del archivo", nodo.lineno < 120)
            for cadena in (
                "await model.ainvoke",
                'return {"messages": [response]}',
                "try:",
                "Salida truncada",
            ):
                _ok(f"dentro de call_model: {cadena}", cadena in cuerpo)
            return
    _ok("existe call_model", False)


def _chequear_cadenas() -> None:
    _ok("sin TODO", "TODO" not in FUENTE)
    arbol = ast.parse(FUENTE)
    _ok("main.py parsea", isinstance(arbol, ast.Module))
    for cadena in CADENAS:
        _ok(f"cadena {cadena}", cadena in FUENTE)
    _chequear_cuerpo(arbol)
    lineas = FUENTE.splitlines()
    grafo = next(i for i, linea in enumerate(lineas, start=1) if "StateGraph(AgentState)" in linea)
    _ok("el grafo está junto a call_model", grafo < 130)


def _chequear_demo() -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        asyncio.run(main())
    salida = buffer.getvalue()
    print(salida, end="")
    _ok("paso 1 consulta Montevideo", "location=Montevideo" in salida)
    _ok("paso 2 recuerda Montevideo", "Para mañana en Montevideo" in salida)
    _ok("checkpoint del thread", "thread_id=dev_test_01" in salida)
    _ok("otro hilo no arrastra la ciudad", "Decime la ciudad para consultar el clima." in salida)


if __name__ == "__main__":
    _configure_stdio()
    _chequear_cadenas()
    _chequear_demo()
    print("[OK] validacion")
