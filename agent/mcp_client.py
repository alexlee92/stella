"""
P3.4 — Client MCP (Model Context Protocol) léger pour Stella.

Permet à Stella de se connecter à des serveurs MCP pour accéder à :
- Bases de données (sqlite, postgres via MCP server)
- Systèmes de fichiers distants
- APIs métier
- Outils IDE (VS Code, PyCharm via extension)

Protocole : JSON-RPC 2.0 over HTTP (serveurs MCP HTTP)
ou subprocess (serveurs MCP stdio).

Configuration : settings.toml -> [mcp]
  [[mcp.servers]]
  name = "database"
  type = "http"
  url = "http://localhost:3000"

  [[mcp.servers]]
  name = "filesystem"
  type = "stdio"
  command = ["npx", "@modelcontextprotocol/server-filesystem", "/path"]
"""

from __future__ import annotations

import json
import subprocess
import uuid
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Configuration MCP
# ---------------------------------------------------------------------------


def _load_mcp_servers() -> list[dict]:
    """Charge les serveurs MCP depuis settings.toml."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return []

    try:
        with open("settings.toml", "rb") as f:
            cfg = tomllib.load(f)
        return cfg.get("mcp", {}).get("servers", [])
    except (OSError, Exception):
        return []


def _find_server(name: str) -> Optional[dict]:
    """Retourne la config du serveur MCP par nom."""
    for srv in _load_mcp_servers():
        if srv.get("name") == name:
            return srv
    return None


# ---------------------------------------------------------------------------
# Appel MCP HTTP
# ---------------------------------------------------------------------------


def _call_http(url: str, method: str, params: dict, timeout: int = 30) -> dict:
    """JSON-RPC 2.0 POST vers un serveur MCP HTTP."""
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params,
    }
    try:
        resp = requests.post(
            f"{url.rstrip('/')}/rpc",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return {"ok": False, "error": data["error"]}
        return {"ok": True, "result": data.get("result")}
    except requests.Timeout:
        return {"ok": False, "error": f"Timeout ({timeout}s) calling {url}"}
    except requests.ConnectionError:
        return {"ok": False, "error": f"Cannot connect to MCP server at {url}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Appel MCP stdio (subprocess)
# ---------------------------------------------------------------------------


def _call_stdio(
    command: list[str], method: str, params: dict, timeout: int = 30
) -> dict:
    """JSON-RPC 2.0 via stdin/stdout d'un processus MCP."""
    payload = (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": method,
                "params": params,
            }
        )
        + "\n"
    )

    try:
        proc = subprocess.run(
            command,
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            return {"ok": False, "error": proc.stderr[:500]}
        data = json.loads(proc.stdout.strip())
        if "error" in data:
            return {"ok": False, "error": data["error"]}
        return {"ok": True, "result": data.get("result")}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Timeout ({timeout}s)"}
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"Invalid JSON from MCP server: {exc}"}
    except FileNotFoundError:
        return {"ok": False, "error": f"Command not found: {command[0]}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def mcp_call(
    server: str,
    method: str,
    params: Optional[dict] = None,
    timeout: int = 30,
) -> dict:
    """
    Appelle une méthode sur un serveur MCP configuré.

    Args:
        server: Nom du serveur MCP (défini dans settings.toml [[mcp.servers]]).
        method: Méthode MCP à appeler (ex: "tools/list", "tools/call", "resources/read").
        params: Paramètres de la méthode.
        timeout: Délai maximal en secondes.

    Returns:
        {"ok": bool, "result": Any} ou {"ok": False, "error": str}
    """
    params = params or {}
    srv = _find_server(server)
    if srv is None:
        return {
            "ok": False,
            "error": (
                f"MCP server '{server}' not found in settings.toml. "
                "Add [[mcp.servers]] with name, type (http|stdio), and url or command."
            ),
        }

    srv_type = srv.get("type", "http")
    if srv_type == "http":
        url = srv.get("url", "")
        if not url:
            return {"ok": False, "error": f"MCP server '{server}': missing 'url'"}
        return _call_http(url, method, params, timeout=timeout)
    elif srv_type == "stdio":
        command = srv.get("command")
        if not command:
            return {"ok": False, "error": f"MCP server '{server}': missing 'command'"}
        if isinstance(command, str):
            command = command.split()
        return _call_stdio(command, method, params, timeout=timeout)
    else:
        return {
            "ok": False,
            "error": f"Unknown MCP server type: {srv_type} (use 'http' or 'stdio')",
        }


def list_mcp_servers() -> list[dict]:
    """Retourne la liste des serveurs MCP configurés avec leur statut."""
    servers = _load_mcp_servers()
    result = []
    for srv in servers:
        info = {
            "name": srv.get("name", "unnamed"),
            "type": srv.get("type", "http"),
            "url_or_cmd": srv.get("url") or str(srv.get("command", "")),
        }
        result.append(info)
    return result


def mcp_tools_list(server: str) -> dict:
    """Liste les outils disponibles sur un serveur MCP."""
    return mcp_call(server, "tools/list")


def mcp_tool_call(server: str, tool_name: str, arguments: dict) -> dict:
    """Appelle un outil spécifique d'un serveur MCP."""
    return mcp_call(server, "tools/call", {"name": tool_name, "arguments": arguments})


def format_mcp_result(result: dict, max_chars: int = 2000) -> str:
    """Formate le résultat d'un appel MCP pour affichage dans le contexte de l'agent."""
    if not result.get("ok"):
        return f"[mcp error] {result.get('error', 'unknown error')}"
    raw = result.get("result")
    if raw is None:
        return "[mcp] No result"
    if isinstance(raw, str):
        return raw[:max_chars]
    return json.dumps(raw, ensure_ascii=False, indent=2)[:max_chars]
