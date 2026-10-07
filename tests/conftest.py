import os

os.environ.setdefault("OPENAI_API_KEY", "sk-solo-tests")

import main  # noqa: E402
from tests.modelo_local import ModeloClimaLocal  # noqa: E402

main.model = ModeloClimaLocal()
