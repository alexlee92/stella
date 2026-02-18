# Stella Agent

Stella est un agent de code local (type Codex/Claude Code) qui tourne avec Ollama + DeepSeek.

## Prerequis

- Python 3.11+
- Ollama accessible depuis Windows: `http://localhost:11434`
- Modeles:
  - `deepseek-coder:6.7b`
  - `nomic-embed-text`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Optionnel:

```bash
copy .env.example .env
```

La config principale est dans `settings.toml`.
Les variables d'environnement surchargent ce fichier.

## Architecture rapide

- `main.py`: CLI
- `agent/auto_agent.py`: boucle autonome (tool-use)
- `agent/memory.py`: index vectoriel chunked + persistant
- `agent/tooling.py`: outils securises read/search/tests/commandes
- `agent/patcher.py`: patch, backup, rollback, transaction multi-fichiers
- `agent/quality.py`: pipeline quality (format/lint/tests)
- `agent/chat_session.py`: chat continu + historique
- `agent/pr_ready.py`: branche + commit + resume diff

## Workflow quotidien

1. Indexer le projet:

```bash
python main.py index
```

2. Poser une question:

```bash
python main.py ask "Explique l'architecture du projet"
```

3. Proposer une modification sans appliquer:

```bash
python main.py review main.py "Refactorise la fonction principale"
```

4. Appliquer une modification:

```bash
python main.py apply main.py "Ajoute une gestion d'erreur"
```

5. Annuler une modif (backup):

```bash
python main.py undo main.py
```

## Mode autonome

Plan (1 decision seulement):

```bash
python main.py plan "Ameliore la robustesse de l'agent"
```

Execution multi-etapes:

```bash
python main.py run "Ameliore la robustesse de l'agent" --steps 10 --apply
```

## Mode chat continu

```bash
python main.py chat --apply
```

Commandes du chat:

- `/run <goal>`: lancer l'agent autonome
- `/plan <goal>`: voir la prochaine decision
- `/decisions`: voir les decisions recentes
- `/exit`: quitter

## PR-ready

Apres un run reussi, prepare une branche + commit + resume diff:

```bash
python main.py pr-ready "amelioration robustesse agent"
```

Avec valeurs explicites:

```bash
python main.py pr-ready "amelioration robustesse agent" --branch feature/agent-robustesse --message "feat: improve agent robustness"
```

Sorties generees:

- Resume console (etat branche/commit/diff)
- Fichier markdown PR: `.stella/last_pr.md` (titre + description)

Si le dossier courant n'est pas un repo git, la commande echoue avec un message d'action:

- `git init` ou cloner un repo, puis relancer `pr-ready`

## Evaluation continue

Executer la suite d'evaluation:

```bash
python main.py eval
```

Rapport ecrit dans:

- `eval/last_report.json`

## CI locale rapide

```bash
python main.py ci
```

## Securite

- Ecriture limitee au `PROJECT_ROOT`
- Commandes shell autorisees uniquement via whitelist (`pytest`, `python -m pytest`, `python -m black`, `python -m ruff`)
- Rollback automatique si echec quality pipeline apres application de patch
- Mode `DRY_RUN=true` pour simuler les operations destructives

## Logs et donnees session

- Historique chat: `.stella/session_history.jsonl`
- Events agent: `.stella/agent_events.jsonl`
- Index memoire: `.stella/memory/`

## Depannage

1. API Ollama inaccessible:

```bash
curl http://localhost:11434/api/tags
```

2. Reindex forcé:

```bash
python main.py index --rebuild
```

3. Si `run --apply` rollback souvent:
- Ajuster `FORMAT_COMMAND`, `LINT_COMMAND`, `TEST_COMMAND` dans `settings.toml`
- Verifier que ces commandes passent manuellement

## Commandes principales

```bash
python main.py -h
python main.py index --rebuild
python main.py ask "..."
python main.py review <file> "..."
python main.py apply <file> "..."
python main.py undo <file>
python main.py plan "..."
python main.py run "..." --steps 10 --apply
python main.py chat --apply
python main.py map
python main.py eval
python main.py ci
python main.py pr-ready "..."
```
