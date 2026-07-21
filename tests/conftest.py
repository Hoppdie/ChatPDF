"""Test configuration for ChatPDF.

``app.py`` imports Streamlit and the heavy ML stack at module scope. Pulling
sentence-transformers/torch/faiss into CI would cost minutes of install time
and buys nothing for testing the pure text-processing logic.

We therefore stub only the modules that are genuinely absent in a test
environment, and only when they are absent -- never shadowing a real
langchain subpackage, since langchain imports its own internals eagerly.
"""
import importlib.util
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _noop(*args, **kwargs):
    return None


def _missing(name):
    """True if `name` cannot be imported in this environment."""
    if name in sys.modules:
        return False
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError, ModuleNotFoundError):
        return True


def _install_stub(name, **attrs):
    """Register a stand-in module only if the real one is unavailable."""
    if not _missing(name):
        return
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


# --- Streamlit: must merely survive import and attribute access ---
if _missing("streamlit"):
    streamlit = types.ModuleType("streamlit")
    for attr in (
        "set_page_config", "write", "header", "text_input", "subheader",
        "file_uploader", "button", "spinner",
    ):
        setattr(streamlit, attr, _noop)
    streamlit.session_state = {}
    sys.modules["streamlit"] = streamlit

# --- python-dotenv: optional at test time ---
_install_stub("dotenv", load_dotenv=_noop)

# --- Embedding backend: heavy, and never invoked by these tests ---
_install_stub("sentence_transformers", SentenceTransformer=object)
