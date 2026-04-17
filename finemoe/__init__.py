__version__ = "0.0.1"

__all__ = ["MoE", "OffloadEngine", "__version__"]


def __getattr__(name):
    if name == "MoE":
        from finemoe.entrypoints import MoE
        return MoE
    if name == "OffloadEngine":
        from finemoe.runtime import OffloadEngine
        return OffloadEngine
    raise AttributeError(name)
