import uuid

from langchain_core.messages import AIMessage, ToolMessage


class ModeloClimaLocal:
    """ainvoke de prueba. Lee el historial que le pasa el grafo. No llama a la red."""

    async def ainvoke(self, messages):
        ultimo = messages[-1]
        if isinstance(ultimo, ToolMessage):
            return _cerrar(messages, str(ultimo.content))
        if _pide_clima(messages):
            ciudad = _ciudad(messages)
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


def _ciudad(messages) -> str | None:
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


def _cerrar(messages, observacion: str) -> AIMessage:
    ciudad = _ciudad(messages) or "ese lugar"
    humano = _ultimo_humano(messages).lower()
    if "mañana" in humano or "manana" in humano:
        return AIMessage(
            content=f"Para mañana en {ciudad} el parte disponible es el de hoy: {observacion}"
        )
    return AIMessage(content=f"En {ciudad}: {observacion}")
