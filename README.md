# Agente de clima con ciclo ReAct y checkpoint

El grafo alterna entre el nodo `agent` y el nodo `tools` hasta que el modelo deja de pedir herramientas. `MemorySaver` guarda el estado después de cada nodo. Un segundo `ainvoke` con el mismo `thread_id` vuelve a ver la ciudad del primer mensaje.

La demo no llama a la API. `usar_modelo_local()` deja en `model` un sustituto con la misma `ainvoke`. El grafo, `ToolNode`, `tools_condition` y el checkpointer son los de LangGraph. Para usar `ChatOpenAI.bind_tools` de verdad: `CODER_USAR_OPENAI=1` y `OPENAI_API_KEY`.

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

3. Demo (no pide clave):

**Windows (PowerShell):**

```powershell
python main.py
```

**Linux/macOS (bash/zsh):**

```bash
python main.py
```

Deberías ver `tool_calls: get_weather location=Montevideo` en el paso 1 y en el paso 2, `Para mañana en Montevideo` y `checkpoint thread_id=dev_test_01`. El otro hilo (`dev_test_02`) responde `Decime la ciudad para consultar el clima.` Si el segundo turno no nombra Montevideo, imprime `Error controlado: ...` y sale con código 1.

## Archivos del repositorio

| Artefacto | Dónde está |
|-----------|------------|
| `AgentState` con `Annotated[list[BaseMessage], add_messages]` | `main.py` |
| `ClimaInput` y `get_weather` | `main.py` |
| `tool_node = ToolNode(tools)` | `main.py` |
| `ChatOpenAI(...).bind_tools(tools)` | `main.py` |
| `async def call_model` | `main.py` |
| `StateGraph`: `START -> agent`, `tools_condition`, `tools -> agent` | `main.py` |
| `memory = MemorySaver()` y `compile(checkpointer=memory)` | `main.py` |
| `ModeloClimaLocal` (demo sin red) | `main.py` |
| Dos `ainvoke` con `thread_id` `dev_test_01` | `main.py` → `main` |
| Chequeo offline | `validacion.py` |
| Tests | `tests/` |
| `pytest.ini` | `testpaths = tests` · `asyncio_mode = auto` · `addopts = -ra -q` |
| `requirements.txt` | `langgraph`, `langchain-core`, `langchain-openai`, `pydantic`, `pytest`, `pytest-asyncio`, `pytest-mock` |

## Comportamiento

| Requisito | Cómo se cumple | Evidencia |
|-----------|----------------|-----------|
| Historial acumulativo | `messages: Annotated[list[BaseMessage], add_messages]` | `test_add_messages_concatena` · `test_mismo_thread_id_recuerda_la_ubicacion` |
| Ciclo ReAct | `agent` pide la herramienta, `tools` ejecuta `get_weather`, el resultado vuelve a `agent`, y termina cuando no hay `tool_calls` | `test_ciclo_react_termina_sin_mas_herramientas` · `test_grafo_tiene_nodos_y_checkpointer` |
| Persistencia | `MemorySaver` y el mismo `thread_id` en los dos `ainvoke` | `test_mismo_thread_id_recuerda_la_ubicacion` · `evidencias/03-demo.txt` |
| Aislamiento | Otro `thread_id` no ve Montevideo | `test_otro_thread_id_no_arrastra_la_ciudad` |
| Reanudación | Si `call_model` falla, `next` queda en `agent`. Se sigue con `ainvoke(None)` y el mismo `thread_id` | `test_timeout_no_borra_el_checkpoint` |

## Contrato de la herramienta

Queda en el docstring de `get_weather`:

- **Nombre:** `get_weather`.
- **Descripción:** consulta el clima actual de una ciudad. Sirve si preguntan el clima, la temperatura o el pronóstico. No sirve para cálculos, precios ni charla general.
- **Esquema de Entrada:** `ClimaInput`. `location` es texto de 1 a 80 caracteres, sin espacios solos y sin campos extra.
- **Esquema de Salida:** un `str`. Montevideo devuelve `Nublado, 18°C, viento del sur`. Una ciudad fuera de la tabla devuelve `Soleado, 22°C`.

## Manejo de errores personalizados

`call_model` no reintenta. La excepción sale como `ErrorDelAgente` y el checkpoint del `thread_id` queda en el último paso que sí terminó. El mensaje humano de ese turno ya está guardado y `state.next` es `("agent",)`. Para seguir, `await app.ainvoke(None, config)` con el mismo `thread_id`. Reenviar el mismo mensaje lo duplica, porque `add_messages` ya lo concatenó.

### Validación Pydantic

**Mensaje:** `Error de validación: location: Value error, location no puede estar vacía. Reformulá location: tiene que ser el nombre de la ciudad.`

Si `location` es un entero: `Input should be a valid string`. Si falta: `Field required`. Si pasa de 80 caracteres: `String should have at most 80 characters`. Un campo que no está en el esquema, por ejemplo `unidad`: `Error de validación: unidad: Extra inputs are not permitted`.

**Cómo reproducirlo:** `ClimaInput(location="   ")` o `get_weather.func("   ")`. En el grafo, un `tool_calls` con `location` en blanco vuelve como observación del rol `tool` (el ciclo sigue) y el texto es el de arriba.

**Test:** `tests/test_grafo.py::test_clima_input_rechaza_location_en_blanco` · `test_clima_input_rechaza_entero_y_campo_extra` · `test_get_weather_func_valida_espacios` · `test_location_en_blanco_vuelve_como_observacion`

### 401 / key

**Mensaje:** `401/key: clave rechazada o ausente. Detalle: Unauthorized: invalid API key No reintentes: corregí la clave y reanudá con ainvoke(None) y el mismo thread_id.`

**Cómo reproducirlo:** sustituir `model` por uno cuya `ainvoke` lance `FalloSimulado(401, "Unauthorized: invalid API key")` y llamar `ainvoke`. Hay una sola llamada.

**Test:** `tests/test_grafo.py::test_401_no_reintenta`

### 429 / cuota

**Mensaje:** `429/cuota: el modelo rechazó el turno por cuota o rate limit. Detalle: rate limit / quota exceeded Reintentá más tarde con ainvoke(None) y el mismo thread_id.`

**Cómo reproducirlo:** `FalloSimulado(429, "rate limit / quota exceeded")` en `model.ainvoke`. Una sola llamada. Más tarde, `ainvoke(None)` con el mismo `thread_id`.

**Test:** `tests/test_grafo.py::test_429_un_solo_intento`

### red / timeout

**Mensaje:** `red/timeout: no se pudo hablar con el modelo. Detalle: Request timed out talking to the model Reanudá con ainvoke(None) y el mismo thread_id. No reenvíes el mensaje.`

**Cómo reproducirlo:** completar un turno sobre Montevideo, después hacer fallar `ainvoke` con `TimeoutError("Request timed out talking to the model")` en el segundo mensaje del mismo hilo. El checkpoint conserva Montevideo y `next` es `("agent",)`. `ainvoke(None)` cierra el segundo turno y la respuesta nombra Montevideo. Un `ConnectionError` usa el mismo texto.

**Test:** `tests/test_grafo.py::test_timeout_no_borra_el_checkpoint`

### Salida truncada

**Mensaje:** `Salida truncada: el modelo cortó la respuesta. Reformulá la consulta y reanudá con ainvoke(None) y el mismo thread_id.`

**Cómo reproducirlo:** `ainvoke` devuelve un `AIMessage` con `response_metadata={"finish_reason": "length"}`. Ese texto no entra al historial. Hay una sola llamada.

**Test:** `tests/test_grafo.py::test_salida_truncada_no_entra_al_historial`

### Error no transitorio

**Mensaje:** `Error del agente: boom Reanudá con ainvoke(None) y el mismo thread_id si el fallo fue transitorio.`

**Cómo reproducirlo:** `model.ainvoke` lanza `RuntimeError("boom")`. Una sola llamada.

**Test:** `tests/test_grafo.py::test_error_no_transitorio_un_solo_intento`

## Evidencias

| Archivo | Qué muestra |
|---------|-------------|
| `evidencias/01-validacion-offline.txt` | `python validacion.py`: cadenas de `main.py` y la demo. |
| `evidencias/02-pytest.txt` | `python -m pytest -v`. |
| `evidencias/03-demo.txt` | `python main.py`. |

## Checklist

- [x] `AgentState` con `Annotated[list[BaseMessage], add_messages]`
- [x] `get_weather` con `ClimaInput` (`location`) y docstring de cuándo usarla
- [x] `tool_node = ToolNode(tools)`
- [x] `ChatOpenAI(...).bind_tools(tools)`
- [x] `call_model` hace `await model.ainvoke(state["messages"])` y devuelve `{"messages": [response]}`
- [x] `START -> agent`, `tools_condition` hacia `tools` o `END`, y `tools -> agent`
- [x] `MemorySaver` pasado a `compile(checkpointer=memory)`
- [x] Dos `ainvoke` con `thread_id` `dev_test_01`: el segundo recuerda Montevideo
- [x] Otro `thread_id` no arrastra la ciudad
- [x] Si el nodo falla, el checkpoint queda y se reanuda con `ainvoke(None)`
- [x] Errores de validación, 401/key, 429/cuota, red/timeout y salida truncada
- [x] `if __name__ == "__main__"` ejecutable, sin API real
- [x] pytest + mocks
- [x] README con una subsección por familia de error
- [x] `evidencias/` con salida real de comandos
