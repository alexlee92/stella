"""
health_check.py â€” VÃ©rification de connectivitÃ© Ollama/routing au dÃ©marrage.
Donne des messages d'aide clairs si un service est inaccessible.
"""

import requests

from agent.config import OLLAMA_BASE_URL, ROUTER_ENABLED, ROUTER_URL


def _check_ollama() -> tuple[bool, str]:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            return True, "OK"
        return False, f"HTTP {resp.status_code}"
    except requests.ConnectionError:
        return False, "connexion refusÃ©e"
    except requests.Timeout:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def _check_routing() -> tuple[bool, str]:
    health_url = ROUTER_URL.replace("/query", "/health")
    try:
        resp = requests.get(health_url, timeout=5)
        if resp.status_code == 200:
            return True, "OK"
        return False, f"HTTP {resp.status_code}"
    except requests.ConnectionError:
        return False, "connexion refusÃ©e"
    except requests.Timeout:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def check_services(verbose: bool = True) -> dict:
    """
    VÃ©rifie la disponibilitÃ© d'Ollama et de routing.
    Affiche des messages d'aide si un service est manquant.
    Retourne {"ollama": bool, "routing": bool, "ok": bool}
    """
    results = {}

    ollama_ok, ollama_msg = _check_ollama()
    results["ollama"] = ollama_ok

    if not ollama_ok and verbose:
        print(f"[!] Ollama inaccessible ({OLLAMA_BASE_URL}) â€” {ollama_msg}")
        print("    Pour dÃ©marrer Ollama :")
        print("      Dans WSL : ollama serve")
        print("      Sur Windows : lancez l'application Ollama")
        print()

    if ROUTER_ENABLED:
        routing_ok, routing_msg = _check_routing()
        results["routing"] = routing_ok
        if not routing_ok and verbose:
            print(f"[!] routing inaccessible ({ROUTER_URL}) â€” {routing_msg}")
            print("    Pour dÃ©marrer le serveur routing :")
            print("      python routing_server.py")
            print("      (depuis le dossier oÃ¹ se trouve routing_server.py)")
            print()
    else:
        results["routing"] = True  # Non requis si dÃ©sactivÃ©

    results["ok"] = results["ollama"] and results.get("routing", True)
    return results


def require_services(skip_check: bool = False) -> bool:
    """
    VÃ©rifie les services et retourne False si les services critiques sont injoignables.
    UtilisÃ© en dÃ©but de commandes qui nÃ©cessitent le LLM.
    """
    if skip_check:
        return True
    result = check_services(verbose=True)
    if not result["ok"]:
        print("[!] Certains services requis sont inaccessibles.")
        print("    DÃ©marrez les services ci-dessus et rÃ©essayez.")
        return False
    return True
