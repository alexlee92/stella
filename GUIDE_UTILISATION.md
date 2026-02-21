# Guide d'Utilisation : Stella Agent

Stella est un agent de code local qui tourne avec les modèles Orisha (deepseek-coder-v2 + qwen2.5-coder via Ollama).

---

## 1. Installation

### Prérequis
- **Python 3.11+**
- **Ollama** : [ollama.com](https://ollama.com)
  - Modèles Orisha : `ollama create Orisha-Ifa1.0 -f Modelfile-Ifa` et `ollama create Orisha-Oba1.0 -f Modelfile-Oba`
  - Modèle d'embedding : `ollama pull nomic-embed-text`

### Installation
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# ou : source .venv/bin/activate  # Linux/Mac

pip install -e .
```

### Première utilisation
```bash
python stella.py init         # indexe le projet + vérifie l'environnement
```

---

## 2. Utilisation — syntaxe simple

**Tu n'as pas besoin de choisir la commande.** Envoie juste ton goal et Stella détecte l'intention :

```bash
python stella.py "ton goal en langage naturel"
```

| Intention détectée | Routé vers | Exemples de déclencheur |
| :--- | :--- | :--- |
| **Question** | réponse directe | `?`, "comment", "qu'est-ce que", "explique", "pourquoi" |
| **Création** | agent autonome | "crée", "génère", "implémente", "nouveau module" |
| **Modification** | fix standard | tout le reste (corrige, ajoute, refactor, bug...) |

### Exemples concrets
```bash
# Question → réponse immédiate (pas de modification)
python stella.py "comment fonctionne le memory index ?"
python stella.py "qu'est-ce que fait la fonction ask_project ?"

# Création → génère les fichiers + installe les dépendances
python stella.py "génère un module auth avec JWT"
python stella.py "crée les tests pour users/api.py"
python stella.py "génère un composant React UserDashboard"

# Modification → lit, patche, applique, teste
python stella.py "corrige les erreurs dans users/api.py"
python stella.py "ajoute la validation email dans users/api.py"
python stella.py "refactorise la boucle dans agent/memory.py"
```

---

## 3. Commandes explicites (avancé)

Si tu veux forcer un mode précis :

| Commande | Usage | Description |
| :--- | :--- | :--- |
| `ask "question"` | lecture seule | Réponse sur le codebase, lit les fichiers mentionnés |
| `fix "description"` | modification | Corrige/améliore, applique les patches |
| `run "goal"` | autonome multi-étapes | Lecture + création + modification + tests |
| `chat` | interactif | Mode conversationnel continu avec `/run`, `/ask`, `/map`... |
| `index` | mémoire | Réindexe tous les fichiers du projet |
| `map` | lecture | Affiche la carte des symboles du projet |
| `doctor` | diagnostic | Vérifie Ollama, modèles, dépendances |

```bash
python stella.py ask "comment est gérée l'authentification ?"
python stella.py fix "corrige le bug de timeout dans agent/auto_agent.py"
python stella.py run "implémente un système de cache Redis" --apply --with-tests
python stella.py chat
python stella.py index --rebuild
```

---

## 4. Préparer une Pull Request

Après avoir fait des modifications avec Stella :

```bash
python stella.py pr-ready "description de la PR"
```

Ce que fait `pr-ready` :
1. Vérifie qu'il y a des changements (`git status`)
2. Lance le quality gate (format + lint + tests)
3. Crée une branche auto-nommée : `agent/YYYYMMDD-description`
4. Commite tous les fichiers modifiés
5. Génère un fichier `.stella/last_pr.md` avec le résumé, le diff et la checklist

```bash
# Options
python stella.py pr-ready "ajout module auth JWT" --branch "feature/auth" --message "feat(auth): add JWT module"
```

---

## 5. Mode Chat

Pour une session interactive (comme un terminal Claude/Codex) :

```bash
python stella.py chat
```

**Commandes disponibles dans le chat :**
- `/run <but>` — lance l'agent autonome
- `/ask <question>` — pose une question sur le code
- `/plan <but>` — affiche le plan sans l'exécuter
- `/status` — état git (branche, fichiers modifiés)
- `/map` — carte des symboles du projet
- `/undo <fichier>` — annule la dernière modification d'un fichier
- `/eval` — lance les tests rapides
- `/help` — liste des commandes

---

## 6. Sécurité et bonnes pratiques

- **Backups automatiques** : chaque fichier modifié est sauvegardé dans `<fichier>.bak_YYYYMMDD_HHMMSS` avant application
- **Rollback** : si les tests échouent, Stella restaure le fichier précédent
- **Undo manuel** : `python stella.py undo chemin/vers/fichier.py`
- **Vérification** : après modification, fais toujours un `git diff` avant de commiter

---

## 7. Résumé — cas d'usage rapide

| Besoin | Commande |
| :--- | :--- |
| Poser une question | `python stella.py "comment fonctionne X ?"` |
| Corriger un bug | `python stella.py "corrige le bug dans fichier.py"` |
| Créer un module | `python stella.py "génère un module de gestion des logs"` |
| Générer des tests | `python stella.py "crée les tests pour mon_module.py"` |
| Nettoyer le code | `python stella.py "passe ruff sur tout le projet et fixe les erreurs"` |
| Préparer une PR | `python stella.py pr-ready "description"` |
| Mode interactif | `python stella.py chat` |
| Diagnostics | `python stella.py doctor` |
