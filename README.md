# Stella Agent

Stella est un agent IA local pour développeurs — similaire à Claude Code, mais qui tourne entièrement sur votre machine avec les modèles Orisha locaux.

---

## Prérequis

- Python 3.11+
- [Ollama](https://ollama.com) accessible sur `http://localhost:11434` (dans WSL ou natif)
- Modèles Orisha chargés dans Ollama :
  - `Orisha-ifa1.0` — architecte / analyste (basé sur deepseek-coder-v2:16b)
  - `Orisha-Oba1.0` — générateur de code (basé sur qwen2.5-coder:14b)
  - `nomic-embed-text` — embeddings pour la mémoire vectorielle

Optionnel (routing automatique) :
- Serveur Flask Orisha (`python orisha_server.py`) sur `http://localhost:5000`

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Vérifier que tout est en ordre :

```bash
python stella.py doctor
```

---

## Démarrage rapide (pour non-experts)

### 1. Initialiser Stella sur votre projet

```bash
python stella.py init
```

Cette commande indexe votre code et configure l'environnement.

### 2. Poser une question sur votre code

```bash
python stella.py ask "Comment fonctionne le module d'authentification ?"
python stella.py ask "Quels fichiers gèrent les routes API ?"
```

### 3. Corriger ou améliorer du code (en langage naturel)

```bash
# Mode sécurisé : propose les changements sans les appliquer
python stella.py fix "Ajouter une gestion d'erreur dans la fonction process_data"

# Mode application directe (après vérification)
python stella.py fix "Corriger le bug de division par zéro" --apply
```

### 4. Voir la carte du projet

```bash
python stella.py map
```

---

## Toutes les commandes

### Questions et analyse

```bash
python stella.py ask "Explique l'architecture du projet"
python stella.py map                                    # carte des fichiers
python stella.py review agent/llm.py "Analyse ce fichier"
```

### Corrections et modifications

```bash
# Proposer une modification (sans appliquer)
python stella.py review agent/llm.py "Améliore la gestion des erreurs"

# Appliquer une modification
python stella.py apply agent/llm.py "Ajoute un timeout"

# Annuler la dernière modification
python stella.py undo agent/llm.py

# Alias simplifié (non-expert)
python stella.py fix "Corriger l'erreur dans agent/llm.py"
python stella.py fix "Corriger l'erreur dans agent/llm.py" --apply
```

### Mode autonome (agent multi-étapes)

```bash
# Planifier une étape (sans exécuter)
python stella.py plan "Améliore la robustesse de l'agent"

# Exécuter en mode autonome (lecture seule, sans modifier les fichiers)
python stella.py run "Analyser et corriger les bugs dans agent/tooling.py" --steps 10

# Exécuter ET appliquer les modifications
python stella.py run "Analyser et corriger les bugs dans agent/tooling.py" --steps 10 --apply

# Mode robuste avec tests
python stella.py run "Corriger le pipeline quality" --apply --with-tests --fix-until-green
```

### Mode chat interactif

```bash
python stella.py chat
python stella.py chat --apply   # applique automatiquement les modifications
```

Commandes disponibles dans le chat :

| Commande | Description |
|----------|-------------|
| `/ask <question>` | Poser une question sur le codebase |
| `/status` | Voir les fichiers modifiés (git) |
| `/map` | Afficher la carte du projet |
| `/undo` | Annuler la dernière modification |
| `/eval` | Lancer les tests rapidement |
| `/help` | Afficher l'aide |
| `/run <objectif>` | Lancer l'agent autonome |
| `/plan <objectif>` | Voir la prochaine décision |
| `/decisions` | Voir les décisions récentes |
| `/exit` | Quitter |

### Tâches de développement guidées

```bash
# Profil safe (lecture seule, pas de modifications)
python stella.py dev-task "Analyser les performances du module mémoire" --profile safe

# Profil standard (modifications avec validation)
python stella.py dev-task "Ajouter des tests unitaires pour agent/risk.py" --profile standard

# Profil agressif (autonome, plus de steps)
python stella.py dev-task "Refactoriser auto_agent.py" --profile aggressive
```

### Qualité et tests

```bash
python stella.py ci                     # CI locale rapide (format + lint + tests)
python stella.py eval                   # Evaluation complète de l'agent
python stella.py eval --limit 3         # Evaluation rapide (3 tâches)
```

### Git et PR

```bash
# Préparer une PR (branche + commit + résumé diff)
python stella.py pr-ready "amélioration robustesse agent"
python stella.py pr-ready "fix: correction bug llm_interface" --branch fix/llm-json
```

---

## Sélection automatique des modèles

Stella choisit automatiquement le bon modèle Orisha selon la tâche :

| Type de tâche | Modèle utilisé | Exemples |
|---------------|----------------|----------|
| `analysis`, `refactor`, `planning`, `json` | Orisha-Ifa1.0 | analyser, expliquer, planifier |
| `debug`, `optimization`, `backend`, `frontend` | Orisha-Oba1.0 | corriger, générer, créer |

La détection fonctionne en **français et en anglais**.

---

## Benchmarks de performance (mesures réelles)

| Tâche | Modèle | Latence |
|-------|--------|---------|
| Question simple | Oba1.0 | ~10s |
| Analyse courte | Ifa1.0 | ~8s |
| Analyse longue | Ifa1.0 | ~27s |
| Génération simple | Oba1.0 | ~15s |
| Génération complexe (classe FastAPI+JWT) | Oba1.0 | ~90s |
| Planning JSON | Ifa1.0 | ~22s |

Stabilité JSON : **100%** sur tous les types de prompts planner.

---

## Architecture

```
stella.py               — CLI principal
agent/
  auto_agent.py         — boucle autonome (planner → critique → action)
  llm_interface.py      — interface LLM avec routing modèle automatique
  memory.py             — index vectoriel (BM25 + cosine) persistant
  tooling.py            — outils sécurisés (read/search/tests)
  patcher.py            — patch transactionnel avec backup/rollback
  quality.py            — pipeline qualité (format/lint/tests)
  health_check.py       — vérification connectivité Ollama/Orisha
  chat_session.py       — chat continu + historique
  pr_ready.py           — branche + commit + résumé diff
bench/
  bench_latency.py      — benchmark latence modèles
  bench_json_stability.py — benchmark stabilité JSON planner
  bench_model_comparison.py — comparaison Ifa1.0 vs Oba1.0
```

---

## Sécurité

- Écriture limitée au dossier du projet (`PROJECT_ROOT`)
- Commandes shell autorisées via whitelist (`pytest`, `black`, `ruff` uniquement)
- Rollback automatique si la quality gate échoue après un patch
- Mode `DRY_RUN=true` pour simuler sans modifier

---

## Données et logs

```
.stella/
  session_history.jsonl   — historique chat
  agent_events.jsonl      — events de l'agent
  memory/                 — index vectoriel persistant
  last_pr.md              — dernière PR préparée
eval/
  last_report.json        — rapport d'évaluation
```

---

## Dépannage

**Ollama inaccessible :**
```bash
# Dans WSL
ollama serve

# Vérifier
curl http://localhost:11434/api/tags
```

**Orisha (Flask) inaccessible :**
```bash
python orisha_server.py
```

**Réindexer le projet :**
```bash
python stella.py index --rebuild
```

**Rollback fréquent après `--apply` :**
- Vérifier que `pytest`, `black`, `ruff` passent manuellement
- Ajuster `FORMAT_COMMAND`, `LINT_COMMAND`, `TEST_COMMAND` dans `settings.toml`

---

## Configuration

La configuration principale est dans `settings.toml`.

Paramètres importants :
- `ORISHA_ENABLED = true` — activer le routing Orisha (Flask requis)
- `context_budget_chars = 8000` — contexte envoyé au modèle
- `AUTO_MAX_STEPS = 15` — nombre d'étapes max en mode autonome
