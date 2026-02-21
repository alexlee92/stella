"""
health_check.py — Vérification de connectivité Ollama/Orisha au démarrage.
Donne des messages d'aide clairs si un service est inaccessible.
"""
import requests

from agent.config import OLLAMA_BASE_URL, ORISHA_ENABLED, ORISHA_URL, REQUEST_TIMEOUT


def _check_ollama() -> tuple[bool, str]:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            return True, "OK"
        return False, f"HTTP {resp.status_code}"
    except requests.ConnectionError:
        return False, "connexion refusée"
    except requests.Timeout:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def _check_orisha() -> tuple[bool, str]:
    health_url = ORISHA_URL.replace("/query", "/health")
    try:
        resp = requests.get(health_url, timeout=5)
        if resp.status_code == 200:
            return True, "OK"
        return False, f"HTTP {resp.status_code}"
    except requests.ConnectionError:
        return False, "connexion refusée"
    except requests.Timeout:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def check_services(verbose: bool = True) -> dict:
    """
    Vérifie la disponibilité d'Ollama et d'Orisha.
    Affiche des messages d'aide si un service est manquant.
    Retourne {"ollama": bool, "orisha": bool, "ok": bool}
    """
    results = {}

    ollama_ok, ollama_msg = _check_ollama()
    results["ollama"] = ollama_ok

    if not ollama_ok and verbose:
        print(f"[!] Ollama inaccessible ({OLLAMA_BASE_URL}) — {ollama_msg}")
        print("    Pour démarrer Ollama :")
        print("      Dans WSL : ollama serve")
        print("      Sur Windows : lancez l'application Ollama")
        print()

    if ORISHA_ENABLED:
        orisha_ok, orisha_msg = _check_orisha()
        results["orisha"] = orisha_ok
        if not orisha_ok and verbose:
            print(f"[!] Orisha inaccessible ({ORISHA_URL}) — {orisha_msg}")
            print("    Pour démarrer le serveur Orisha :")
            print("      python orisha_server.py")
            print("      (depuis le dossier où se trouve orisha_server.py)")
            print()
    else:
        results["orisha"] = True  # Non requis si désactivé

    results["ok"] = results["ollama"] and results.get("orisha", True)
    return results


def require_services(skip_check: bool = False) -> bool:
    """
    Vérifie les services et retourne False si les services critiques sont injoignables.
    Utilisé en début de commandes qui nécessitent le LLM.
    """
    if skip_check:
        return True
    result = check_services(verbose=True)
    if not result["ok"]:
        print("[!] Certains services requis sont inaccessibles.")
        print("    Démarrez les services ci-dessus et réessayez.")
        return False
    return True
