# Audit Complet et Approfondi — Stella

Date: 2026-02-25
Projet audité: `C:\Users\lodon\PycharmProjects\stella`

## Méthodologie

Audit réalisé sur 4 axes:
- Exécution: commandes CLI réelles (`test`, `ci`, `doctor`, `watch`).
- Qualité statique: `ruff check .`, `black --check .`, `bandit -r agent stella.py`.
- Tests: `pytest` et `PYTHONPATH=. pytest -p no:cacheprovider`.
- Revue de code ciblée: `stella.py`, `agent/tooling.py`, `agent/scaffolder.py`, `agent/progress.py`, `scripts_run_ci.py`, `tests/*`.

## Résumé Exécutif

État global: **fonctionnel partiellement**, avec plusieurs défauts bloquants sur l'expérience CLI et la robustesse QA.

Points critiques:
1. `watch` ne fonctionne pas (collision argparse).  
2. `stella.py test` échoue par défaut alors que la suite de tests passe avec configuration correcte (`PYTHONPATH`).
3. L'implémentation `ci` ne correspond pas à la promesse produit (pas de lint/format/tests, seulement `py_compile`).
4. Les garde-fous shell sont plus permissifs que ce qui est annoncé (risque sécurité).

## Constats Détaillés (Priorisés)

### Critique

#### C1 — Commande `watch` cassée (non exécutable en pratique)
- Impact: fonctionnalité annoncée mais inutilisable.
- Cause racine:
  - Le subparser stocke la sous-commande dans `args.command` (`stella.py:124`).
  - L'option `--command` du `watch` réutilise le même nom de destination (`stella.py:291`), écrasant la sous-commande.
  - Le dispatch `if args.command == "watch"` (`stella.py:889`) n'est donc jamais atteint.
- Reproduction:
  - `python -u stella.py watch --pattern "tests/*.py" --command "pytest -q" --interval 1`
  - Résultat observé: sortie vide, watcher jamais lancé.
- Correction recommandée:
  - Renommer l'argument CLI `--command` en `dest="watch_command"`.
  - Utiliser `args.watch_command` dans l'appel `run_watch(...)`.

#### C2 — `stella.py test` en échec par défaut (pipeline QA faux négatif)
- Impact: commande principale de test non fiable; faux signal d'échec.
- Reproduction:
  - `python stella.py test` -> `ModuleNotFoundError: No module named 'agent'`.
  - `PYTHONPATH=. pytest -p no:cacheprovider` -> **84 passed**.
- Analyse:
  - `stella.py test` exécute `pytest -q` (`stella.py:874-878`) sans forcer un mode stable d'import.
  - `tests/test_stella.py` injecte le path (`tests/test_stella.py:13`) mais `tests/test_auto_agent.py` ne le fait pas.
- Correction recommandée:
  - Standardiser sur `python -m pytest -q` dans `TEST_COMMAND` / `run_tests`.
  - Ajouter `conftest.py` (ou config pytest) pour normaliser `sys.path` côté tests.

### Majeur

#### M1 — Commande `ci` incohérente avec la documentation
- Impact: fausse confiance sur la qualité livrée.
- Observé:
  - README annonce: format + lint + tests.
  - Implémentation réelle: seulement `py_compile` sur quelques fichiers (`scripts_run_ci.py:21-31`, appelé par `stella.py:794-798`).
- Risque: du code non formaté/non linté/non testé peut passer `ci`.
- Correction recommandée:
  - Remplacer `scripts_run_ci.py` par une vraie pipeline: `black --check`, `ruff check`, `pytest`.

#### M2 — Whitelist shell trop large vs promesse sécurité
- Impact: surface d'exécution de commandes plus grande qu'annoncée.
- Observé:
  - README indique whitelist limitée (`pytest`, `black`, `ruff`).
  - Code autorise aussi `python`, `pip`, `npm`, `node`, `docker`, `alembic`, etc. (`agent/tooling.py:36-95`).
  - `run_safe_command` utilise `shell=True` (`agent/tooling.py:177-186`), signalé en High par Bandit.
- Risque: exécution de commandes inattendues via agent/planner.
- Correction recommandée:
  - Réduire la whitelist aux commandes strictement nécessaires.
  - Passer en exécution `subprocess.run(list_args, shell=False)`.
  - Ajouter validation forte des arguments.

#### M3 — Génération de templates: collisions de noms de fichiers
- Impact: impossible de générer plusieurs templates Python pour une même entité/dossier.
- Cause:
  - Pour presque tous les templates, nom de sortie = `{name}.py` (`agent/scaffolder.py:1260-1262`).
  - Le second scaffold retourne `Le fichier existe deja` (`agent/scaffolder.py:1268-1271`).
- Reproduction:
  - Génération de 17 templates: la majorité échoue par collision sur `sample.py`.
- Correction recommandée:
  - Préfixer/suffixer par type (`invoice_sample.py`, `rbac_sample.py`, etc.) ou arborescence dédiée par template.

#### M4 — Fonction `progress` inutilisable par défaut
- Impact: commande présente mais sans artefact source.
- Cause:
  - `summarize_progress()` attend `UPGRADE_PLAN_30J.md` (`agent/progress.py:33`) absent du repo.
- Reproduction:
  - `python stella.py progress` -> `Progress unavailable: [Errno 2] No such file or directory`.
- Correction recommandée:
  - Soit ajouter le fichier attendu, soit rendre le chemin configurable avec fallback robuste.

### Modéré

#### O1 — Dette qualité statique importante
- `ruff check .` retourne **18 erreurs** (imports inutilisés, variables inutilisées, f-strings invalides).
- `black --check .` indique un grand nombre de fichiers non formatés (commande timeout après 120s).
- Impact: maintenabilité, bruit PR, baisse de signal QA.

#### O2 — Templates de tests livrent des stubs non actionnables
- Le template `test` génère `assert True  # TODO: implement` (`agent/scaffolder.py:195, 199, 203`).
- Impact: couverture fictive si ces tests sont conservés tels quels.

#### O3 — Risque de bord au chargement de `agent.auth`
- `agent/auth.py` lève immédiatement si `JWT_SECRET_KEY` absent (`agent/auth.py:6-11`).
- Impact: import non robuste pour certains usages/librairie.

### Information

#### I1 — Couverture de tests limitée sur l'empreinte fonctionnelle
- Modules `agent/*.py`: 49.
- Modules importés par tests: 10.
- Implication: beaucoup de fonctionnalités critiques (watcher, scaffolder avancé, migration, mcp, deps, etc.) peu ou pas testées automatiquement.

#### I2 — `bandit` signale 33 issues (5 High)
- High principaux: `shell=True`, `debug=True` en exemple Flask, usage SHA1.
- Certaines alertes sont contextuelles, mais elles doivent être triées et annotées (`# nosec` justifié ou correction).

## Fonctionnalités Absentes ou Incomplètes

1. `watch` non opérationnel (bug critique de parsing CLI).  
2. `progress` dépend d'un fichier de plan absent.  
3. `ci` "réel" (lint/format/tests) absent malgré la promesse documentation.  
4. Génération de tests scaffold non finalisée (stubs TODO).  
5. Gestion robuste de l'environnement de test absente (`stella test` fragile sans `PYTHONPATH`).

## Recommandations Prioritaires (Plan d'Action)

### Sprint 1 (immédiat)
1. Corriger `watch` (`dest=watch_command` + dispatch).  
2. Fiabiliser `test`/`watch` sur `python -m pytest` + normalisation import tests.  
3. Aligner `ci` avec README (black/ruff/pytest) ou corriger README.

### Sprint 2
1. Réduire whitelist shell et retirer `shell=True` quand possible.  
2. Corriger stratégie de nommage des fichiers scaffold.  
3. Ajouter fallback clair pour `progress`.

### Sprint 3
1. Nettoyage `ruff` + formatage global.  
2. Durcir templates de tests (vrais tests minimaux).  
3. Étendre la couverture de tests aux modules non couverts.

## Résultats de Validation Exécutés

- `pytest` -> erreur import `agent` (configuration par défaut).  
- `PYTHONPATH=. pytest -p no:cacheprovider` -> **84 passed**.  
- `ruff check .` -> **18 erreurs**.  
- `black --check .` -> nombreux fichiers à reformater (timeout de la commande).  
- `python stella.py ci` -> passe (mais ne fait que `py_compile`).  
- `python stella.py doctor` -> 9/9 checks OK dans cet environnement.

## Conclusion

Le socle Stella est prometteur, mais il existe des écarts significatifs entre fonctionnalités annoncées et comportement réel, principalement sur la CLI, la chaîne qualité, et la sécurité d'exécution shell. Les correctifs prioritaires sont ciblés et faisables rapidement; ils amélioreront immédiatement la fiabilité utilisateur et la crédibilité du projet.

## Suivi des Correctifs Appliqués (2026-02-25)

- [x] C1 — `watch` corrigé (collision argparse).
Fait:
`stella.py` (`--command` -> `dest="watch_command"`, dispatch `run_watch(... command=args.watch_command ...)`).

- [x] C2 — `stella test` fiabilisé + imports tests stabilisés.
Fait:
`stella.py` (`test` utilise `python -m pytest -q`), ajout de `tests/conftest.py` pour injecter la racine projet dans `sys.path`, alignement config `test_command` dans `settings.toml` et `agent/settings.py`.

- [x] M1 — `ci` alignée avec la promesse produit.
Fait:
`scripts_run_ci.py` exécute maintenant `ruff format --check`, `ruff check`, `pytest -q`.

- [x] M4 — `progress` rendu utilisable sans `UPGRADE_PLAN_30J.md`.
Fait:
`agent/progress.py` ajoute fallback automatique vers `ADAPTATION_PLAN_SINGLE_MODEL.md` puis `AUDIT_COMPLET_STELLA_2026-02-23.md`, avec indication de la source utilisée.

- [x] M3 — collisions de noms `scaffold` corrigées complètement.
Fait:
`agent/scaffolder.py` résout désormais les collisions de façon robuste:
nom initial, puis préfixe template (`<template>_<filename>`), puis suffixes incrémentaux (`_2`, `_3`, ...).

- [x] M2 — durcissement sécurité shell (`shell=True`, whitelist trop large).
Fait:
`agent/tooling.py` exécute désormais les commandes en `shell=False` avec tokenisation explicite (`shlex.split`) et suppression de la whitelist générique `["python"]` trop permissive.

### Validation post-correctifs

- `python stella.py test`: OK, **84 passed**.
- `python -u stella.py watch --pattern "tests/*.py" --command "python -m pytest -q" --interval 1`: démarrage watcher confirmé.
- `python stella.py progress`: OK (source fallback: `ADAPTATION_PLAN_SINGLE_MODEL.md`).
- `scaffold` anti-collision: OK (`sample.py` puis `employee_sample.py` générés dans le même dossier).
- `python stella.py ci`: OK, pipeline verte.

### Validation additionnelle (phase suivante)

- `ruff` dette nettoyée: `python -m ruff check .` -> OK.
- Formatage harmonisé: migration vers `ruff format` pour la CI locale (`scripts_run_ci.py` + `settings.toml`).
- `python stella.py ci` (format + lint + tests): **OK**.
