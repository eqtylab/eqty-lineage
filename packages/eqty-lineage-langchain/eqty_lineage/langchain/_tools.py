"""Capturing tool source at definition time, since callbacks only carry a tool's name."""

import inspect
import logging
from typing import Any, Dict

logger = logging.getLogger("eqty.langgraph")

#: tool name -> Python source captured by @eqty_tool, read by every EqtyCallbackHandler instance.
#: Module-level, so a handler loaded as a separate module object gets its own copy -- see the note in
#: the package README about registering tools against the module that will observe them.
_registered_tool_sources: Dict[str, str] = {}


def eqty_tool(obj: Any) -> Any:
    """Capture a tool function's source code so ``EqtyCallbackHandler`` registers the Tool asset from it.

    Callbacks only receive a tool's *name* at runtime, so the source must be recorded at definition time.
    Returns ``obj`` unchanged and works on either side of LangChain's ``@tool`` decorator::

        @tool
        @eqty_tool
        def search(query: str) -> str: ...
    """
    # when stacked outside @tool, obj is a StructuredTool holding the original fn in .func/.coroutine
    fn = getattr(obj, "func", None) or getattr(obj, "coroutine", None) or obj
    name = getattr(obj, "name", None) or getattr(fn, "__name__", None)
    if name is not None:
        try:
            _registered_tool_sources[name] = inspect.getsource(fn)
        except (OSError, TypeError):
            logger.debug("no source available for tool '%s'", name)
    return obj
