import importlib
import pkgutil
import forge.tools as tools_pkg

TOOLS = {}

def load_tools():
    for module in pkgutil.iter_modules(tools_pkg.__path__):
        name = module.name
        mod = importlib.import_module(f"forge.tools.{name}")

        if hasattr(mod, "run"):
            TOOLS[name] = mod.run

def get_tool(name: str):
    return TOOLS.get(name)
