# Pre-import torch before pytest assertion rewriting begins.
# torch._C (C extension) crashes when loaded inside pytest's exec_module
# context; importing it here (which is not rewritten) caches it in sys.modules
# so all subsequent test-module imports see an already-initialised module.
import torch  # noqa: F401
