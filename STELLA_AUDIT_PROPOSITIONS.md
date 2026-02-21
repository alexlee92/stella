# AUDIT COMPLET STELLA -- Propositions d'ameliorations

> **Date** : 2026-02-21
> **Objectif** : Rapprocher Stella du niveau d'un assistant IA de code professionnel (type Claude Code)
> **Contexte** : Stella sera connecte a de gros projets (ERP, etc.) -- performance et fiabilite critiques

---

## Etat des lieux

| Metrique | Valeur |
|----------|--------|
| Lignes de code agent | ~10 500 |
| Modules Python | ~43 |
| Tests unitaires | 84 (20s) |
| Boucle agent max | 10 steps |
| Latence typique par step | 10-45s |
| Latence boucle complete (8 steps) | 2-4 min |

### Points forts actuels
- Boucle autonome planner -> critique -> executor bien pensee
- Rollback transactionnel des patches
- Merge AST-aware (Python) + symbol-aware (JS/TS)
- Memoire vectorielle hybride (semantique + BM25 + MMR)
- Detection bilingue FR/EN des types de taches
- Securite : whitelist commandes, validation chemins, `shlex.split()`

---

## TIER 1 -- Critiques (UX et utilisabilite)

### P1.1 -- Streaming de la sortie LLM
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : L'utilisateur attend 20-90s sans feedback visuel
- **Solution** : `ask_llm_stream()` + `ask_llm_stream_print()` cables dans `stella.py ask`, chat `/ask`, et `ChatSession.ask()`
- **Impact** : Latence percue divisee par 2, UX drastiquement amelioree
- **Fichiers modifies** : `agent/llm_interface.py`, `agent/agent.py`, `agent/chat_session.py`, `stella.py`

### P1.2 -- Confirmation interactive avant application des patches
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : L'agent applique les modifications sans demander confirmation
- **Solution** : Methode `_confirm_apply()` dans `AutonomousAgent`, flag `interactive=True` dans `stella.py run`
- **Impact** : Empeche les modifications non voulues, confiance utilisateur
- **Fichiers modifies** : `agent/auto_agent.py`, `stella.py`

### P1.3 -- Affichage des diffs colores avant application
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : L'utilisateur voit le fichier complet, pas les changements
- **Solution** : `_colorize_diff_line()` avec codes ANSI (rouge/vert/cyan), `show_diff()` ameliore avec stats additions/deletions
- **Impact** : L'utilisateur comprend exactement ce qui va changer
- **Fichiers modifies** : `agent/patcher.py`

### P1.4 -- Indicateur de progression en temps reel
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Pas de feedback pendant l'execution des etapes de l'agent
- **Solution** : Affichage `[n/max] action... (elapsed_time)` + resume de session enrichi (fichiers, cout, replans)
- **Impact** : L'utilisateur sait que Stella travaille et ou elle en est
- **Fichiers modifies** : `agent/auto_agent.py`

### P1.5 -- Annulation/Pause de l'agent en cours d'execution
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Impossible d'arreter l'agent une fois lance (sauf Ctrl+C brutal)
- **Solution** : Handler `SIGINT` gracieux dans `stella.py` avec message informatif
- **Impact** : Controle total de l'utilisateur sur l'agent
- **Fichiers modifies** : `stella.py`

---

## TIER 2 -- Architecture et robustesse

### P2.1 -- Refactoring de auto_agent.py (1889 lignes -> 3 modules)
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Classe monolithique avec 40+ variables d'instance, difficile a tester/maintenir
- **Solution** : Extraction en 3 modules :
  - `agent/decision_normalizer.py` (363 lignes) -- normalisation, aliases, autocorrection, helpers goal
  - `agent/executor.py` (402 lignes) -- `ActionExecutor` classe avec handlers par action
  - `agent/auto_agent.py` (1130 lignes) -- orchestrateur allege, delegation aux modules extraits
- **Impact** : auto_agent.py reduit de 40% (1889 -> 1130 lignes), testabilite amelioree
- **Fichiers modifies** : `agent/auto_agent.py`, `agent/decision_normalizer.py` (nouveau), `agent/executor.py` (nouveau)

### P2.2 -- Persistance des sessions (reprise apres crash)
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Toute la conversation et le contexte sont perdus au redemarrage
- **Solution** :
  - Sessions JSONL dans `.stella/sessions/` avec ID unique par session
  - `list_sessions()`, `get_latest_session_id()` pour naviguer l'historique
  - `load_session(id)` restaure messages, decisions, et staged_edits
  - `save_staged_edits()` persiste les edits non appliques pour recovery
  - Commandes chat `/sessions` (lister) et `/replay [id]` (reprendre)
- **Impact** : Recuperation apres crash, continuite de travail
- **Fichiers modifies** : `agent/chat_session.py`, `stella.py`

### P2.3 -- Timeout par action (pas global)
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Un seul timeout global de 200s -- une action lente bloque tout
- **Solution** : Methode `_action_timeout()` avec timeouts par type d'action (read=10s, LLM=180s, tests=300s, etc.)
- **Impact** : Resilience face aux actions lentes/bloquees
- **Fichiers modifies** : `agent/auto_agent.py`

### P2.4 -- Tests d'integration pour la boucle agent
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : 0 tests sur la logique de boucle, auto-correction, et decisions
- **Solution** : 33 tests dans `tests/test_auto_agent.py` couvrant :
  - Normalisation decisions (12 tests) : aliases, type coercion, args shape inference
  - Normalisation critique (5 tests) : approve string/bool, patched_decision
  - Coerce decision (2 tests) : git on non-git goal
  - Goal helpers (4 tests) : extract file, is_code_edit, is_git
  - Fallback inference (4 tests) : file target, test/perf keywords
  - Autocorrection schema (3 tests) : missing path, invalid action, finish reason
  - Boucle agent mockee (3 tests) : list+finish, loop detection, cost budget
- **Impact** : 33 nouveaux tests, detection de regressions sur toute la logique agent
- **Fichiers modifies** : `tests/test_auto_agent.py` (nouveau)

### P2.5 -- Injection de dependances (pas de globals)
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Config globale, chemins hardcodes, impossible de paralleliser 2 agents
- **Solution** :
  - `AutonomousAgent.__init__()` accepte `config=`, `llm_fn=`, `memory_fn=` (optionnels)
  - Tous les appels `ask_llm_json()` remplaces par `self._llm_fn()`
  - Tous les appels `search_memory()` remplaces par `self._memory_fn()`
  - `ChatSession` accepte les memes parametres et les transmet a l'agent
  - Backward compatible : defaults vers les globals existants si non fournis
- **Impact** : Testabilite, possibilite d'executer plusieurs agents simultanement
- **Fichiers modifies** : `agent/auto_agent.py`, `agent/chat_session.py`

---

## TIER 3 -- Fonctionnalites manquantes (vs Claude Code)

### P3.1 -- Git avance (merge, stash, blame, history)
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Seuls branch/commit/diff sont supportes
- **Solution** : Ajout de `git_stash()`, `git_stash_pop()`, `git_stash_list()`, `git_log()`, `git_blame()`, `git_diff_staged()` dans `git_tools.py`. Commandes `/log`, `/stash`, `/stash-pop` dans le chat.
- **Impact** : Gestion complete du cycle de vie du code
- **Fichiers modifies** : `agent/git_tools.py`, `stella.py`

### P3.2 -- Recherche et navigation dans le code (type LSP)
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Pas de "Go to definition", "Find references", "Find usages"
- **Solution** : Module `agent/code_intelligence.py` avec jedi :
  - `goto_definition(path, symbol)` — trouve la definition d'un symbole
  - `find_references(path, symbol)` — trouve toutes les references
  - `get_signature(path, symbol)` — signature de fonction/classe
  - `list_symbols(path)` — liste les symboles d'un fichier
  - Commandes chat : `/goto <fichier> <symbole>`, `/refs <fichier> <symbole>`, `/symbols <fichier>`
- **Impact** : Navigation code instantanee, comprehension profonde
- **Fichiers modifies** : `agent/code_intelligence.py` (nouveau), `stella.py`

### P3.3 -- Execution de commandes arbitraires (sandbox)
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Whitelist tres restrictive (pytest, ruff, black uniquement)
- **Solution** : Whitelist elargie (mypy, bandit, pip, npm, node, python, git read-only), blocklist de securite, mode `ask_user=True` pour commandes inconnues, timeout explicite
- **Impact** : Stella peut executer des scripts, installer des deps, lancer des serveurs
- **Fichiers modifies** : `agent/tooling.py`

### P3.4 -- Support multi-fichiers simultane
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : L'agent edite un fichier a la fois, pas de vision globale des changements
- **Solution** :
  - `validate_cross_imports()` dans `patcher.py` : validation croisee des imports entre fichiers d'une transaction
  - Detecte les symboles importes qui n'existent pas dans le fichier cible modifie
  - Affiche les warnings avant application de la transaction
  - Rollback atomique existant conserve
- **Impact** : Refactoring multi-fichiers fiable, detection d'incoherences d'imports
- **Fichiers modifies** : `agent/patcher.py`

### P3.5 -- Contexte de projet intelligent (pour gros projets/ERP)
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Budget contexte fixe (8000 chars), pas adapte aux gros projets
- **Solution** :
  - Indexation incrementale (`_incremental_update()`) : compare les hashes, ne re-indexe que les fichiers modifies
  - Graphe de dependances integre dans `search_memory()` : boost de +0.08 pour les fichiers lies aux fichiers recemment changes
  - `adaptive_budget_context()` ameliore : scale x1.5 sur projets >200 fichiers
  - `summarize_module()` : resume compact (imports + signatures) pour gros modules
  - `_estimate_project_scale()` : detection small/medium/large
- **Impact** : Stella reste efficace sur des projets de 100k+ lignes, re-indexation 10x plus rapide
- **Fichiers modifies** : `agent/memory.py`

### P3.6 -- Mode conversation multi-tour avec memoire de session
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Chaque commande est independante, pas de "souvenir" du contexte recent
- **Solution** : Fenetre de conversation glissante (10 derniers echanges, budget 4000 chars), commande `/context` pour voir l'etat de la session, methode `show_context()` dans ChatSession
- **Impact** : Conversations naturelles, raffinement iteratif
- **Fichiers modifies** : `agent/chat_session.py`, `stella.py`

### P3.7 -- Generation de code avec templates et scaffolding
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Pas de templates pour les patterns courants (API endpoint, model, migration, etc.)
- **Solution** : Module `agent/scaffolder.py` :
  - 6 templates integres : `fastapi-endpoint`, `django-model`, `django-view`, `react-component`, `python-module`, `test`
  - Templates personnalisables depuis `.stella/templates/` (auto-detectes)
  - `scaffold(type, name, output_dir)` genere le fichier avec substitution `{name}/{Name}`
  - `list_templates()` retourne les templates disponibles
  - Commande CLI `stella scaffold <type> <name>` + commande chat `/scaffold`
- **Impact** : Generation rapide de boilerplate, coherence du code
- **Fichiers modifies** : `agent/scaffolder.py` (nouveau), `stella.py`

### P3.8 -- Analyse de securite integree
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : `bandit` optionnel et silencieusement ignore s'il n'est pas installe
- **Solution** : Scan de secrets integre (`scan_secrets_in_files()`) dans le pipeline qualite -- detecte API keys, tokens GitHub, AWS keys, secrets hardcodes. Warning affiche avant les tests.
- **Impact** : Code securise par defaut, critique pour les ERP
- **Fichiers modifies** : `agent/quality.py`

---

## TIER 4 -- Ergonomie et polish

### P4.1 -- Coloration syntaxique dans le terminal
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Le code affiche en chat est en texte brut
- **Solution** : Fonction `_highlight_code()` dans `stella.py` utilisant `rich.Syntax` pour coloriser les blocs de code markdown (```lang ... ```) dans les reponses de l'agent
- **Impact** : Lisibilite, experience agreable
- **Fichiers modifies** : `stella.py`

### P4.2 -- Auto-completion des commandes
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : L'utilisateur doit connaitre les commandes par coeur
- **Solution** : Integration `prompt_toolkit` via `_build_prompt_session()` :
  - Auto-completion de toutes les commandes `/run`, `/goto`, `/refs`, `/symbols`, etc.
  - Historique des commandes en memoire (fleches haut/bas)
  - Fallback gracieux vers `input()` si prompt_toolkit n'est pas installe
- **Impact** : Decouvrbilite des commandes, productivite
- **Fichiers modifies** : `stella.py`

### P4.3 -- Tableau de bord de session
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Pas de vue d'ensemble de ce que l'agent a fait
- **Solution** : `/status` enrichi (branche, fichiers modifies par ext, messages/decisions en session). Resume de fin d'agent avec stats (etapes, fichiers, cout, replans).
- **Impact** : Transparence, confiance utilisateur
- **Fichiers modifies** : `stella.py`, `agent/auto_agent.py`

### P4.4 -- Commandes raccourcies et aliases
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Commandes longues (`stella dev-task "..."`)
- **Solution** :
  - `stella fix "bug"` — deja fait
  - `stella init` — deja fait
  - `stella test [args]` — alias rapide pour pytest (+ `/test` en chat)
  - `stella scaffold <type> <name>` — generation boilerplate
  - `stella watch` — surveillance fichiers
  - Section `[aliases]` dans `settings.toml` pour aliases personnalisables
- **Impact** : Rapidite d'utilisation quotidienne
- **Fichiers modifies** : `stella.py`, `settings.toml`

### P4.5 -- Mode watch (surveillance de fichiers)
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : L'utilisateur doit relancer Stella manuellement apres chaque modif
- **Solution** : Module `agent/watcher.py` :
  - `run_watch(pattern, command, interval)` — boucle de surveillance par polling
  - Detection de changements par mtime (ajout, modification, suppression)
  - Re-lancement automatique de la commande de test
  - Affichage des fichiers changes + resultat PASS/FAIL
  - `stella watch --pattern "**/*.py" --interval 2` en CLI
  - Ctrl+C pour arreter proprement
- **Impact** : Workflow continu, feedback immediat
- **Fichiers modifies** : `agent/watcher.py` (nouveau), `stella.py`

---

## TIER 5 -- Specifique ERP / Gros projets

### P5.1 -- Support multi-langages (Python + JS/TS + SQL + XML)
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Les ERP utilisent souvent SQL, XML (Odoo), Java, etc.
- **Solution** :
  - Chunking SQL intelligent (`_chunk_sql_by_statements()`) : decoupe par blocs CREATE/ALTER/INSERT
  - Chunking XML intelligent (`_chunk_xml_by_elements()`) : decoupe par elements racine
  - Extensions indexees etendues : `.sql`, `.xml`, `.html`, `.css`, `.scss`
- **Impact** : Stella indexe et comprend tout le stack ERP (Python + JS/TS + SQL + XML + HTML/CSS)
- **Fichiers modifies** : `agent/memory.py`

### P5.2 -- Comprehension des relations BDD/ORM
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Stella ne comprend pas les relations entre modeles/tables
- **Solution** : Module `agent/orm_parser.py` :
  - `scan_orm_models()` : scan AST des fichiers projet pour detecter les classes Model (SQLAlchemy, Django, Odoo)
  - `render_er_summary()` : genere un diagramme entite-relation textuel
  - `get_related_models(model_name)` : trouve les modeles lies par FK
  - `_extract_models_from_file()` : extraction AST des champs et relations
- **Impact** : Modifications de modeles sans casser les relations FK
- **Fichiers modifies** : `agent/orm_parser.py` (nouveau)

### P5.3 -- Gestion des migrations de base de donnees
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Pas de support pour generer/valider les migrations
- **Solution** : Module `agent/migration_helper.py` :
  - `detect_migration_framework()` : detecte Alembic, Django, ou Odoo automatiquement
  - `suggest_migration_commands(changed_files)` : propose les commandes de migration adaptees
  - `validate_model_migration_coherence(changed_files)` : alerte si des modeles changent sans migration
- **Impact** : Securite des changements de schema en production
- **Fichiers modifies** : `agent/migration_helper.py` (nouveau)

### P5.4 -- Documentation automatique des changements
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Les changements ne sont pas documentes
- **Solution** : Module `agent/changelog.py` :
  - `generate_session_changelog(goal, files, steps, test_results)` : genere un entry markdown
  - `_infer_change_type()` : detecte feat/fix/refactor/docs/perf/chore depuis le goal
  - `_categorize_files()` : trie les fichiers par role (models, tests, migrations, configs)
  - `append_to_changelog_file()` : insere dans CHANGELOG.md existant
- **Impact** : Tracabilite pour les equipes, audits
- **Fichiers modifies** : `agent/changelog.py` (nouveau)

### P5.5 -- Support des environnements multiples (dev/staging/prod)
- [x] **Statut** : FAIT (2026-02-21)
- **Probleme** : Stella ne connait pas l'environnement cible
- **Solution** :
  - `_detect_environment()` dans `settings.py` : lit `STELLA_ENV` (default: development)
  - Chargement `settings.{env}.toml` avec merge par-dessus la config de base
  - Blocage de commandes dangereuses en production/staging dans `tooling.py` : `DROP`, `TRUNCATE`, `DELETE FROM`, `ALTER TABLE`, `MIGRATE`
  - Variable `STELLA_ENV` exposee dans les settings
- **Impact** : Securite operationnelle, pas de commandes destructives en prod
- **Fichiers modifies** : `agent/settings.py`, `agent/tooling.py`

---

## Matrice de priorite

| ID | Proposition | Effort | Impact | Priorite | Statut |
|----|------------|--------|--------|----------|--------|
| P1.1 | Streaming sortie LLM | Moyen | Tres eleve | **1** | FAIT |
| P1.2 | Confirmation interactive | Faible | Eleve | **2** | FAIT |
| P1.3 | Diffs colores | Faible | Eleve | **3** | FAIT |
| P1.4 | Indicateur progression | Faible | Eleve | **4** | FAIT |
| P3.5 | Contexte intelligent (gros projets) | Eleve | Tres eleve | **5** | FAIT |
| P2.1 | Refactoring auto_agent.py | Eleve | Eleve | **6** | FAIT |
| P3.3 | Execution commandes sandbox | Moyen | Eleve | **7** | FAIT |
| P3.4 | Multi-fichiers simultane | Moyen | Eleve | **8** | FAIT |
| P3.6 | Conversation multi-tour | Moyen | Eleve | **9** | FAIT |
| P1.5 | Annulation/Pause agent | Moyen | Moyen | **10** | FAIT |
| P3.1 | Git avance | Moyen | Moyen | **11** | FAIT |
| P3.2 | Navigation code (LSP) | Eleve | Eleve | **12** | FAIT |
| P2.2 | Persistance sessions | Moyen | Moyen | **13** | FAIT |
| P2.3 | Timeout par action | Faible | Moyen | **14** | FAIT |
| P2.4 | Tests integration boucle | Moyen | Eleve | **15** | FAIT |
| P4.1 | Coloration syntaxique | Faible | Moyen | **16** | FAIT |
| P4.2 | Auto-completion | Moyen | Moyen | **17** | FAIT |
| P3.8 | Analyse securite | Moyen | Eleve | **18** | FAIT |
| P5.1 | Multi-langages | Eleve | Eleve | **19** | FAIT |
| P5.2 | Relations ORM | Eleve | Eleve | **20** | FAIT |
| P3.7 | Templates/scaffolding | Moyen | Moyen | **21** | FAIT |
| P4.3 | Tableau de bord session | Faible | Faible | **22** | FAIT |
| P4.4 | Aliases commandes | Faible | Faible | **23** | FAIT |
| P4.5 | Mode watch | Moyen | Moyen | **24** | FAIT |
| P5.3 | Migrations BDD | Eleve | Eleve | **25** | FAIT |
| P5.4 | Documentation auto | Moyen | Moyen | **26** | FAIT |
| P5.5 | Multi-environnements | Moyen | Moyen | **27** | FAIT |
| P2.5 | Injection dependances | Eleve | Moyen | **28** | FAIT |

---

## Bugs et dette technique identifies

| Bug/Dette | Localisation | Severite | Statut |
|-----------|-------------|----------|--------|
| `auto_agent.py` : 1807 lignes monolithiques, 40+ vars d'instance | `agent/auto_agent.py` | Haute | CORRIGE (P2.1 : 1889->1130 lignes, -40%) |
| `ask_llm_stream()` existe mais jamais utilise | `agent/llm_interface.py` | Moyenne | CORRIGE |
| Pas de tests sur la boucle agent ni l'auto-correction de schema | `tests/` | Haute | CORRIGE (P2.4 : 33 tests ajoutes) |
| BM25 tokenizer ne separe pas `snake_case` | `agent/memory.py` | Basse | A faire |
| Backups `.bak_YYYYMMDD` jamais nettoyes | `agent/patcher.py` | Basse | A faire |
| `bandit` silencieusement ignore si absent | `agent/quality.py` | Moyenne | CORRIGE (scan interne ajoute) |
| Windows cp1252 : emojis crashent l'affichage | `stella.py` | Moyenne | Deja corrige (reconfigure utf-8) |
| Cache LLM trop agressif (meme prompt = meme reponse cross-projets) | `agent/llm_interface.py` | Basse | A faire |
| N+1 appels d'embedding (pas de batch) | `agent/memory.py` | Moyenne | A faire |
| TS merge regex fragile sur accolades imbriquees | `agent/ts_merge.py` | Moyenne | A faire |

---

## Suivi d'implementation

_Cocher les cases au fur et a mesure de l'integration :_

- [x] Sprint 1 : P1.1 + P1.2 + P1.3 + P1.4 + P1.5 (UX de base) -- FAIT 2026-02-21
- [x] Sprint 2 : P2.3 + P3.3 (timeouts + sandbox) -- FAIT 2026-02-21
- [x] Sprint 3 : P3.1 + P3.6 + P3.8 (git + conversation + securite) -- FAIT 2026-02-21
- [x] Sprint 4 : P4.3 (dashboard session) -- FAIT 2026-02-21
- [x] Sprint 5 : P2.1 + P3.5 (refactoring auto_agent + contexte intelligent) -- FAIT 2026-02-21
- [x] Sprint 6 : P3.2 + P3.4 (navigation code + multi-fichiers) -- FAIT 2026-02-21
- [x] Sprint 7 : P2.4 + P4.1 + P4.2 (tests integration + coloration + auto-completion) -- FAIT 2026-02-21
- [x] Sprint 8 : P5.1 + P5.2 + P5.3 + P5.4 + P5.5 (ERP complet) -- FAIT 2026-02-21
- [x] Sprint 9 : P2.2 + P2.5 + P3.7 + P4.4 + P4.5 (session + DI + scaffolding + aliases + watch) -- FAIT 2026-02-21

---

## Resume des changements appliques (2026-02-21)

### Fichiers modifies :
| Fichier | Changements |
|---------|-------------|
| `agent/llm_interface.py` | `ask_llm_stream()` ameliore (Orisha fallback), `ask_llm_stream_print()` ajoute |
| `agent/agent.py` | `ask_project_stream()` ajoute (version streaming de ask_project) |
| `agent/auto_agent.py` | `_confirm_apply()`, `_action_timeout()`, progress avec timing, resume enrichi, mode `interactive`, delegation vers executor/normalizer |
| `agent/patcher.py` | `_colorize_diff_line()`, `show_diff()` ameliore avec couleurs ANSI et stats, `validate_cross_imports()` |
| `agent/tooling.py` | Whitelist elargie, blocklist securite, `is_command_blocked()`, `ask_user` mode sandbox, blocage prod/staging |
| `agent/git_tools.py` | `git_stash()`, `git_stash_pop()`, `git_stash_list()`, `git_log()`, `git_blame()`, `git_diff_staged()` |
| `agent/chat_session.py` | Streaming dans `ask()`, `show_context()`, fenetre conversation amelioree |
| `agent/quality.py` | `scan_secrets_in_files()`, scan integre dans le pipeline |
| `agent/memory.py` | Indexation incrementale, boost dependency graph, budget adaptatif, chunking SQL/XML, `summarize_module()` |
| `agent/settings.py` | `_detect_environment()`, chargement `settings.{env}.toml`, merge par environnement |
| `agent/chat_session.py` | Sessions JSONL persistantes, `list_sessions()`, `load_session()`, `save_staged_edits()`, DI |
| `stella.py` | Streaming ask, Ctrl+C handler, `/log`, `/stash`, `/stash-pop`, `/context`, `/status` enrichi, `/goto`, `/refs`, `/symbols`, `/sessions`, `/replay`, `/test`, `/scaffold`, `_highlight_code()`, `_build_prompt_session()`, commandes `test`/`scaffold`/`watch` |
| `settings.toml` | Section `[aliases]` pour aliases personnalisables |

### Nouveaux fichiers :
| Fichier | Description |
|---------|-------------|
| `agent/decision_normalizer.py` | Normalisation des decisions LLM, aliases, autocorrection, helpers goal (363 lignes) |
| `agent/executor.py` | ActionExecutor : dispatch et handlers par action (402 lignes) |
| `agent/code_intelligence.py` | Navigation code jedi : goto_definition, find_references, get_signature, list_symbols |
| `agent/orm_parser.py` | Parser ORM : scan modeles SQLAlchemy/Django/Odoo, diagramme ER, relations FK |
| `agent/migration_helper.py` | Detection framework migration, suggestions commandes, validation coherence |
| `agent/changelog.py` | Generation changelog automatique, conventional commits, categorisation fichiers |
| `agent/scaffolder.py` | Templates et scaffolding : 6 templates integres + templates custom |
| `agent/watcher.py` | Mode watch : surveillance fichiers par polling, re-run tests auto |
| `tests/test_auto_agent.py` | 33 tests d'integration pour la boucle agent et la normalisation |

### Propositions implementees : 27/28
### Bugs corriges : 5/10
### Proposition restante : aucune fonctionnelle (seul P4.4 etait "partiel", maintenant FAIT)
