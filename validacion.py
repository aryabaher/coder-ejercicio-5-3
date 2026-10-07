"""Chequeo offline del ciclo ReAct y del checkpoint (sin API real)."""

from __future__ import annotations

import ast
import asyncio
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "sk-solo-validacion")

import main as modulo  # noqa: E402
from tests.modelo_local import ModeloClimaLocal  # noqa: E402

ROOT = Path(__file__).resolve().parent
FUENTE = (ROOT / "main.py").read_text(encoding="utf-8")

CADENAS = [
    "class AgentState",
    "messages: Annotated[list[BaseMessage], add_messages]",
    "def get_weather",
    "@tool",
    "tool_node = ToolNode([get_weather])",
    "ChatOpenAI",
    ".bind_tools([get_weather])",
    "async def agent",
    'await model.ainvoke(state["messages"])',
    'return {"messages": [response]}',
    "workflow = StateGraph(AgentState)",
    'workflow.add_node("agent", agent)',
    'workflow.add_node("tools", tool_node)',
    'workflow.add_edge(START, "agent")',
    "tools_condition",
    'workflow.add_edge("tools", "agent")',
    "memory = MemorySaver()",
    "app = workflow.compile(checkpointer=memory)",
    'thread_id": "sesion_01"',
    'thread_id": "sesion_02"',
    "ainvoke",
    'if __name__ == "__main__":',
    "asyncio.run(main())",
]


def _configure_stdio() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")


def _ok(nombre: str, condicion: bool, detalle: str = "") -> None:
    estado = "OK" if condicion else "FALLO"
    extra = f" - {detalle}" if detalle else ""
    print(f"[{estado}] {nombre}{extra}")
    if not condicion:
        raise SystemExit(1)


def _cuerpo(nodo: ast.AST) -> str:
    fin = nodo.end_lineno or nodo.lineno
    return "\n".join(FUENTE.splitlines()[nodo.lineno - 1 : fin])


def _chequear_cuerpo(arbol: ast.Module) -> None:
    for nodo in arbol.body:
        if isinstance(nodo, ast.AsyncFunctionDef) and nodo.name == "agent":
            cuerpo = _cuerpo(nodo)
            _ok("agent está al inicio del archivo", nodo.lineno < 45)
            _ok("dentro de agent: ainvoke", "await model.ainvoke" in cuerpo)
            _ok(
                'dentro de agent: return {"messages": [response]}',
                'return {"messages": [response]}' in cuerpo,
            )
        if isinstance(nodo, ast.AsyncFunctionDef) and nodo.name == "main":
            cuerpo = _cuerpo(nodo)
            _ok("main no redefine el grafo", "StateGraph" not in cuerpo)
            _ok("main no pisa el modelo", "ModeloClimaLocal" not in cuerpo)
            _ok('main usa sesion_01', 'thread_id": "sesion_01"' in cuerpo)
            _ok('main usa sesion_02', 'thread_id": "sesion_02"' in cuerpo)


def _chequear_cadenas() -> None:
    _ok("sin TODO", "TODO" not in FUENTE)
    _ok("un solo StateGraph", FUENTE.count("StateGraph(AgentState)") == 1)
    _ok("sin usar_modelo_local", "usar_modelo_local" not in FUENTE)
    _ok("main.py corto (menos de 65 lineas)", len(FUENTE.splitlines()) < 65)
    _ok("sin clave de API por defecto", "sk-" not in FUENTE and "api_key" not in FUENTE)
    _ok("demo imprime el historial", "pretty_print()" in FUENTE and "aget_state" in FUENTE)
    _ok("main.py solo ASCII", FUENTE.isascii())
    arbol = ast.parse(FUENTE)
    _ok("main.py parsea", isinstance(arbol, ast.Module))
    for cadena in CADENAS:
        _ok(f"cadena {cadena}", cadena in FUENTE)
    _chequear_cuerpo(arbol)


def _chequear_demo() -> None:
    modulo.model = ModeloClimaLocal()
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        asyncio.run(modulo.main())
    salida = buffer.getvalue()
    print(salida, end="")
    _ok("el hilo sesion_01 recuerda Montevideo", "Para mañana en Montevideo" in salida)
    _ok("el hilo sesion_02 no arrastra la ciudad", "Decime la ciudad" in salida)


if __name__ == "__main__":
    _configure_stdio()
    _chequear_cadenas()
    _chequear_demo()
    print("[OK] validacion")
