# Stella Agent

Stella est un agent IA local pour développeurs — similaire à Claude Code, mais qui tourne entièrement sur votre machine avec Ollama et un modèle principal unique.

---

## Prérequis

- Python 3.11+
- [Ollama](https://ollama.com) accessible sur `http://localhost:11434` (dans WSL ou natif)
- Modèles Ollama chargés :
  - `qwen2.5-coder:14b-instruct-q5_K_M` — modèle principal Stella
  - `nomic-embed-text` — embeddings pour la mémoire vectorielle

```bash
ollama pull qwen2.5-coder:14b-instruct-q5_K_M
ollama pull nomic-embed-text
```

---

## Installation

```bash
cd /path/to/stella
python -m venv .venv

# Linux / macOS / WSL
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\activate

pip install -e .
```

Vérifier que tout est en ordre :

```bash
python stella.py doctor
```

---

## Utiliser Stella sur votre propre projet

Stella s'utilise **depuis le dossier de n'importe quel projet**. Deux étapes suffisent.

### Étape 1 — Pointer Stella vers votre projet

Editez `settings.toml` dans le dossier Stella :

```toml
[project]
root = "/path/to/your/project"   # chemin absolu de votre projet
```

Ou lancez directement depuis le dossier cible (Stella détecte `.` comme racine) :

```bash
cd /path/to/your/project
python /path/to/stella/stella.py init
```

> `init` indexe le code, configure git si absent, installe les outils de qualité.

### Étape 2 — Adapter le quality gate

Dans `settings.toml`, remplacez les commandes par celles de **votre stack** :

```toml
[quality]
# Python
format_command = "python -m ruff format ."
lint_command   = "python -m ruff check ."
test_command   = "python -m pytest -q"

# Node / TypeScript
# format_command = "prettier --write ."
# lint_command   = "eslint ."
# test_command   = "npm test"
```

Si votre projet n'a pas encore de tests :
```toml
test_command = ""   # désactive la gate de test
```

### Alias pratique (optionnel)

Pour éviter de taper le chemin complet, ajoutez dans votre shell (`~/.bashrc`, `~/.zshrc`, ou PowerShell profile) :

```bash
# Linux / macOS / WSL
alias stella="python /path/to/stella/stella.py"
```

```powershell
# PowerShell
function stella { python C:\path\to\stella\stella.py @args }
```

Ensuite depuis n'importe quel projet :

```bash
cd /path/to/your/project
stella init
stella ask "Comment fonctionne le module de paiement ?"
stella run "Ajoute la pagination au endpoint /api/products" --apply
```

---

## Démarrage rapide

### 1. Initialiser

```bash
python stella.py init
```

### 2. Poser une question sur votre code

```bash
python stella.py ask "Comment fonctionne le module d'authentification ?"
python stella.py ask "Quels fichiers gèrent les routes API ?"
```

### 3. Corriger ou améliorer du code

```bash
# Propose les changements sans les appliquer (mode sécurisé)
python stella.py fix "Ajouter une gestion d'erreur dans la fonction process_data"

# Applique directement
python stella.py fix "Corriger le bug de division par zéro" --apply
```

### 4. Mode simple (goal direct)

```bash
python stella.py "votre objectif en langage naturel"
```

Routage automatique :
- Question (`comment`, `pourquoi`, `?`) → `ask`
- Création (`crée`, `génère`, `implémente`) → `run --apply`
- Le reste (correction/amélioration) → `fix`

---

## Toutes les commandes

### Questions et analyse

```bash
python stella.py ask "Explique l'architecture du projet"
python stella.py map                                        # carte des fichiers et symboles
python stella.py review src/api.py "Analyse ce fichier"
```

### Corrections et modifications

```bash
# Proposer une modification (sans appliquer)
python stella.py review src/api.py "Améliore la gestion des erreurs"

# Appliquer une modification
python stella.py apply src/api.py "Ajoute un timeout"

# Annuler la dernière modification
python stella.py undo src/api.py

# Alias simplifié
python stella.py fix "Corriger l'erreur dans src/api.py"
python stella.py fix "Corriger l'erreur dans src/api.py" --apply
```

### Mode autonome (agent multi-étapes)

```bash
# Voir la prochaine décision sans exécuter
python stella.py plan "Améliore la robustesse de l'agent"

# Exécuter en lecture seule (sans modifier les fichiers)
python stella.py run "Analyser les bugs dans src/tooling.py" --steps 10

# Exécuter ET appliquer les modifications
python stella.py run "Corriger les bugs dans src/tooling.py" --steps 10 --apply

# Mode robuste : boucle jusqu'à ce que les tests passent
python stella.py run "Corriger le pipeline qualité" --apply --with-tests --fix-until-green
```

### Mode chat interactif

```bash
python stella.py chat
python stella.py chat --apply   # applique automatiquement les modifications
```

Commandes disponibles dans le chat :

| Commande | Description |
|----------|-------------|
| `/run <objectif>` | Lancer l'agent autonome |
| `/ask <question>` | Poser une question sur le codebase |
| `/plan <objectif>` | Voir la prochaine décision |
| `/status` | Voir les fichiers modifiés (git) |
| `/map` | Afficher la carte du projet |
| `/undo <fichier>` | Annuler la dernière modification |
| `/eval` | Lancer les tests rapidement |
| `/decisions` | Voir les décisions récentes |
| `/stash` / `/stash-pop` | Sauvegarder/restaurer le travail en cours |
| `/scaffold <type> <nom>` | Générer un fichier depuis un template |
| `/help` | Afficher l'aide complète |
| `/exit` | Quitter |

### Génération de fichiers (scaffold)

```bash
python stella.py scaffold fastapi-endpoint UserEndpoint
python stella.py scaffold django-model Product
python stella.py scaffold react-component LoginForm
python stella.py scaffold test MyService
```

Types disponibles : `fastapi-endpoint`, `django-model`, `django-view`, `react-component`, `python-module`, `celery-task`, `test`.

### Tâches de développement guidées

```bash
# Profil safe (lecture seule, analyse uniquement)
python stella.py dev-task "Analyser les performances du module mémoire" --profile safe

# Profil standard (modifications avec validation)
python stella.py dev-task "Ajouter des tests unitaires pour src/risk.py" --profile standard

# Profil agressif (autonome, plus de steps)
python stella.py dev-task "Refactoriser le module d'authentification" --profile aggressive
```

### Qualité et tests

```bash
python stella.py ci                      # CI locale rapide (format + lint + tests)
python stella.py test                    # alias rapide pour pytest -q
python stella.py eval                    # évaluation complète de l'agent
python stella.py eval --limit 3          # évaluation rapide (3 tâches)
python stella.py eval --tasks eval/tasks_code_edit_prod.json --min-generation-quality 75
```

Le rapport d'évaluation inclut un score `code_edit_generation_quality_score`
(syntaxe, absence de stubs/TODO, imports wildcard, annotations, présence de tests).

### Surveillance fichiers

```bash
python stella.py watch                   # relance pytest à chaque modification
python stella.py watch --pattern "src/**/*.py" --command "npm test"
```

### Git et PR

```bash
python stella.py pr-ready "amélioration robustesse agent"
python stella.py pr-ready "fix: correction bug llm_interface" --branch fix/llm-json
```

---

## Comportements importants

| Comportement | Détail |
|---|---|
| Sans `--apply` | Stella **stage** les changements en mémoire, ne touche rien sur disque |
| Avec `--apply` | Écrit sur disque + lance la quality gate, **rollback automatique** si ça casse |
| Ctrl+C | Sauvegarde automatique dans `.stella/staged_recovery.json`, rechargé au prochain run avec le même goal |
| Boucles | Détection et arrêt automatiques (re-lectures légitimes tolerées) |
| Sécurité | Toute écriture est limitée à `PROJECT_ROOT`, commandes shell via whitelist |

---

## Architecture

```
stella.py                   — CLI principal
agent/
  auto_agent.py             — boucle autonome (planner → critique → action)
  planner.py                — génération des décisions LLM
  critic.py                 — validation des décisions
  executor.py               — exécution des actions
  loop_controller.py        — détection de boucles et budget
  replan.py                 — replanification après échec
  llm_interface.py          — interface LLM (Ollama / OpenAI)
  memory.py                 — index vectoriel (BM25 + cosine) persistant
  tooling.py                — outils sécurisés (read/search/tests/web)
  patcher.py                — patch transactionnel avec backup/rollback
  quality.py                — pipeline qualité (format/lint/tests)
  scaffolder.py             — templates de fichiers
  chat_session.py           — chat continu + historique
  pr_ready.py               — branche + commit + résumé diff
  global_memory.py          — mémoire cross-sessions
bench/
  bench_latency.py          — benchmark latence modèles
  bench_json_stability.py   — benchmark stabilité JSON planner
```

---

## Configuration (`settings.toml`)

```toml
[models]
main  = "qwen2.5-coder:14b-instruct-q5_K_M"   # modèle principal
embed = "nomic-embed-text"                      # embeddings mémoire

[ollama]
base_url        = "http://localhost:11434"
request_timeout = 200                           # secondes

[routing]
enabled = false    # toujours false en mode standard (appel Ollama direct)

[project]
root = "."         # "." = dossier courant, ou chemin absolu vers votre projet

[agent]
auto_max_steps    = 10    # étapes max en mode autonome
auto_test_command = "python -m pytest -q"
dry_run           = false # true = simule sans modifier

[quality]
format_command = "python -m ruff format ."
lint_command   = "python -m ruff check ."
test_command   = "python -m pytest -q"

[memory]
context_budget_chars = 8000   # contexte envoyé au modèle par fichier
```

---

## Sécurité

- Écriture strictement limitée à `PROJECT_ROOT` (path traversal bloqué)
- Commandes shell autorisées via whitelist (`pytest`, `ruff`, `npm`, `docker`, etc.)
- Rollback automatique si la quality gate échoue après un patch
- Résultats web sanitizés avant injection dans le prompt LLM
- Mode `dry_run = true` pour simuler sans aucune modification

---

## Données et logs

```
.stella/
  session_history.jsonl     — historique chat
  agent_events.jsonl        — events de l'agent (décisions, erreurs, replans)
  memory/                   — index vectoriel persistant
  staged_recovery.json      — sauvegarde automatique sur Ctrl+C
  last_pr.md                — dernière PR préparée
eval/
  last_report.json          — rapport d'évaluation
```

---

## Dépannage

**Ollama inaccessible :**
```bash
# Démarrer Ollama (dans WSL ou terminal natif)
ollama serve

# Vérifier
curl http://localhost:11434/api/tags
```

**Réindexer le projet** (après ajout de nombreux fichiers) :
```bash
python stella.py index --rebuild
```

**Rollback fréquent après `--apply` :**
- Vérifier que la quality gate passe manuellement : `ruff format --check . && ruff check . && pytest -q`
- Ajuster `format_command`, `lint_command`, `test_command` dans `settings.toml` selon votre stack
- Désactiver temporairement les gates qui échouent en vidant la commande : `test_command = ""`

**Récupérer un travail interrompu (Ctrl+C) :**
```bash
# Relancer avec le même goal — Stella recharge automatiquement staged_recovery.json
python stella.py run "votre goal d'origine" --apply
```
