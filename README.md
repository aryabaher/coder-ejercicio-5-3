# Agente de clima con ciclo ReAct y checkpoint

El grafo alterna entre el nodo `agent` y el nodo `tools` hasta que el modelo deja de pedir herramientas. `MemorySaver` guarda el estado después de cada nodo. Un segundo `ainvoke` con el mismo `thread_id` vuelve a ver la ciudad del primer mensaje.

`model` es `ChatOpenAI(...)` con `bind_tools([get_weather])`. El grafo se arma una sola vez, a nivel de módulo, y `main()` solo lo ejecuta.

Los tests sustituyen `model` por un doble en memoria. `pytest` no llama a la API.

## Quick path

1. Entorno e instalación. Tests: `pytest`, `pytest-asyncio`, `pytest-mock`. Un `pip install -r requirements.txt` alcanza.

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux/macOS (bash/zsh):**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Chequeo offline y tests:

```
python validacion.py
python -m pytest -v
```

`python main.py` usa `ChatOpenAI` y necesita `OPENAI_API_KEY`. El chequeo offline y pytest no la usan.

## Archivos del repositorio

| Artefacto | Dónde está |
|-----------|------------|
| `AgentState` con `Annotated[list[BaseMessage], add_messages]` | `main.py` |
| `@tool` `get_weather` | `main.py` |
| `tool_node = ToolNode([get_weather])` | `main.py` |
| `ChatOpenAI(...)` + `bind_tools([get_weather])` | `main.py` |
| `async def agent` | `main.py` |
| `StateGraph`: `START -> agent`, `tools_condition`, `tools -> agent` | `main.py` |
| `memory = MemorySaver()` y `compile(checkpointer=memory)` | `main.py` |
| Dos `ainvoke` con `thread_id` `sesion_01` | `main.py` → `main` |
| Doble de `ainvoke` para los tests | `tests/modelo_local.py` |
| Chequeo offline | `validacion.py` |
| Tests | `tests/` |
| `pytest.ini` | `testpaths = tests` · `asyncio_mode = auto` · `addopts = -ra -q` |
| `requirements.txt` | `langgraph`, `langchain-core`, `langchain-openai`, `pydantic`, `pytest`, `pytest-asyncio`, `pytest-mock` |

## Comportamiento

| Requisito | Cómo se cumple | Evidencia |
|-----------|----------------|-----------|
| Historial acumulativo | `messages: Annotated[list[BaseMessage], add_messages]` | `test_add_messages_concatena` · `test_mismo_thread_id_recuerda_la_ubicacion` |
| Ciclo ReAct | `agent` pide la herramienta, `tools` ejecuta `get_weather`, el resultado vuelve a `agent`, y termina cuando no hay `tool_calls` | `test_ciclo_react_termina_sin_mas_herramientas` · `test_grafo_tiene_nodos_y_checkpointer` |
| Persistencia | `MemorySaver` y el mismo `thread_id` `sesion_01` en los dos `ainvoke` | `test_mismo_thread_id_recuerda_la_ubicacion` · `evidencias/03-demo.txt` |
| Aislamiento | Otro `thread_id` no ve Montevideo | `test_otro_thread_id_no_arrastra_la_ciudad` |
| Modelo | `ChatOpenAI.bind_tools`. El grafo no se vuelve a crear dentro de `main` | `test_el_modelo_del_modulo_es_chatopenai` |

## Contrato de la herramienta

Queda en el docstring de `get_weather`:

- **Nombre:** `get_weather`.
- **Descripción:** consulta el clima actual de una ciudad. Sirve si preguntan el clima, la temperatura o el pronóstico. No sirve para cálculos, precios ni charla general.
- **Entrada:** `location`, el nombre de la ciudad.
- **Salida:** un `str`. Montevideo devuelve `Nublado, 18 C`. Una ciudad fuera de la tabla devuelve `Soleado, 22 C`.

## Tests sin API

`tests/conftest.py` asigna `main.model` a `ModeloClimaLocal` antes de cada test. `agent` sigue haciendo `await model.ainvoke(state["messages"])`. `validacion.py` hace lo mismo y después corre `main()`: `sesion_01` imprime `Para mañana en Montevideo...` y `sesion_02` pide la ciudad.

## Evidencias

| Archivo | Qué muestra |
|---------|-------------|
| `evidencias/01-validacion-offline.txt` | `python validacion.py`: cadenas de `main.py` y la demo con el doble. |
| `evidencias/02-pytest.txt` | `python -m pytest -v`. |
| `evidencias/03-demo.txt` | La misma demo offline que `validacion.py`. |

## Checklist

- [x] `AgentState` con `Annotated[list[BaseMessage], add_messages]`
- [x] `@tool` `get_weather` con docstring de cuándo usarla
- [x] `tool_node = ToolNode([get_weather])`
- [x] `ChatOpenAI(...)` con `bind_tools([get_weather])`, y `main()` no lo reemplaza
- [x] `agent` hace `await model.ainvoke(state["messages"])` y devuelve `{"messages": [response]}`
- [x] Un solo `StateGraph`: `add_node("agent", agent)`, `tools_condition`, `tools -> agent`
- [x] `MemorySaver` pasado a `compile(checkpointer=memory)`
- [x] Dos `ainvoke` en `sesion_01` y un hilo aparte `sesion_02`
- [x] `if __name__ == "__main__"` ejecutable
- [x] pytest con doble de `ainvoke`, sin API
- [x] `evidencias/` con salida real de comandos
