"""AI layer: lazy speech recognition engine and model management (M3).

Heavy dependencies (``faster_whisper``, ``ctranslate2``) are imported inside the
functions that create a model, so importing this package never loads CUDA,
CTranslate2 or a Whisper model. Import submodules explicitly
(``from app.ai.asr import get_engine``).
"""
