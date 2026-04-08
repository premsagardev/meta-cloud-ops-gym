# server/app.py
# Validator-required entry point. Imports the FastAPI app from the root
# server.py so all logic stays in one place and nothing is duplicated.

import sys
import os

# Ensure the repo root is on sys.path so `import server` resolves to
# the root-level server.py, not this package.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import importlib.util as _ilu

# Load root server.py explicitly by file path to avoid the package
# shadowing it (since this directory is also named "server").
_spec = _ilu.spec_from_file_location("_root_server", os.path.join(_root, "server.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export the FastAPI app so uvicorn can target server.app:app
app = _mod.app


def main() -> None:
    """Entry point wired up in [project.scripts] as `server`."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
