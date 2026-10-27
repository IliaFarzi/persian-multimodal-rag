import threading

from embedding_runtime import get_embedding_runtime, release_embedding_runtime
from vgen_runtime import get_vgen_runtime, release_vgen_runtime


_RUNTIME_SWITCH_LOCK = threading.RLock()


def activate_vgen_runtime(runtime=None):
    with _RUNTIME_SWITCH_LOCK:
        release_embedding_runtime()
        runtime = runtime if runtime is not None else get_vgen_runtime()
        runtime.prepare()
        return runtime


def activate_embedding_runtime():
    with _RUNTIME_SWITCH_LOCK:
        release_vgen_runtime()
        runtime = get_embedding_runtime()
        runtime.prepare()
        return runtime


def activate_qa_runtimes():
    with _RUNTIME_SWITCH_LOCK:
        vgen_runtime = get_vgen_runtime()
        vgen_runtime.prepare()
        embedding_runtime = get_embedding_runtime()
        embedding_runtime.prepare()
        return vgen_runtime, embedding_runtime


def release_all_model_memory():
    with _RUNTIME_SWITCH_LOCK:
        release_vgen_runtime()
        release_embedding_runtime()
