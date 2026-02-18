# Plan Upgrade 30 Jours (Local 7B)

Ce document sert de suivi d'avancement pour rendre l'agent local plus fiable et plus utilisable au quotidien, malgre la contrainte modele 7B.

## Contexte

- Modele local limite a 7B (pas d'upgrade taille modele)
- Strategie: gagner en architecture, outillage, validation, evaluation

## Objectif global

Atteindre un agent "tres fiable pour usage personnel" avec:
- pipeline robuste
- faibles erreurs de planification
- patchs plus surs
- eval continue

## Statut global

- Date de demarrage: 2026-02-17
- Avancement global: 67%
- Etat: En cours

---

## Semaine 1 - Fiabilite decisionnelle

### Taches

- [x] Ajouter un schema JSON strict par action outil
- [x] Ajouter validation dure du schema avant execution
- [x] Ajouter self-check en 2 passes (plan puis critique)
- [x] Renforcer detection de boucle (action + args + sortie)
- [x] Journaliser les echecs classes (parse/tool/test/rollback)

### Livrables

- [x] `agent/` validation schema centralisee
- [x] logs d'erreurs structurees consultables
- [x] reduction des erreurs JSON planner

### Statut semaine 1

- Progression: 100%
- Notes:
  - `agent/action_schema.py` ajoute schema + validation
  - `agent/auto_agent.py` ajoute critique 2 passes + loop guards renforces
  - `agent/event_logger.py` ajoute `log_failure(...)`

---

## Semaine 2 - Qualite code et patchs

### Taches

- [x] Introduire patch AST-aware Python (quand possible)
- [x] Ajouter tests cibles auto selon fichiers modifies
- [x] Pipeline qualite 2 niveaux (rapide/complet)
- [x] Ajouter score de risque patch

### Livrables

- [x] patch plus fin que remplacement de fichier complet
- [x] meilleure precision des validations
- [x] metrique risque visible avant apply

### Statut semaine 2

- Progression: 100%
- Notes:
  - `agent/ast_merge.py` + integration `agent/patcher.py`
  - `agent/test_selector.py` + quality fast mode cible
  - `agent/quality.py` ajoute `run_quality_pipeline(mode=...)` et `run_quality_gate(...)`
  - `agent/risk.py` + affichage CLI dans `review/apply`

---

## Semaine 3 - RAG optimise pour 7B

### Taches

- [x] Chunking par symboles (fonction/classe) en priorite
- [x] Re-ranking hybride (vectoriel + lexical/BM25-like)
- [x] Context budgeter (top-N utile + resume)
- [x] Cache des requetes frequentes

### Livrables

- [x] contexte plus pertinent par requete
- [x] baisse du bruit injecte au modele
- [x] latence reduite sur requetes repetitives

### Statut semaine 3

- Progression: 100%
- Notes:
  - `agent/memory.py` passe en chunking symbolique Python avec fallback fenetre
  - reranking hybride semantic + BM25-lite + boost chemin
  - budgeteur de contexte: `budget_context(...)`
  - cache LRU des requetes memoire
  - `agent/agent.py` utilise budget contextuel pour `ask` et `propose`

---

## Semaine 4 - UX + Evaluation continue

### Taches

- [x] Etendre benchmark a 50 taches reelles
- [x] Mesurer KPI automatiquement apres eval
- [x] Ajouter mode `doctor` (sanity check environnement)
- [x] Enrichir `pr-ready` avec checklist validation

### Livrables

- [x] `eval/tasks.json` enrichi
- [x] rapport KPI stable et comparable
- [x] diagnostic rapide avant execution agent

### Statut semaine 4

- Progression: 100%
- Notes:
  - `eval/tasks.json` passe a 50 scenarios (ask+auto)
  - `agent/eval_runner.py` calcule et ecrit `.stella/kpi_snapshot.json`
  - `agent/doctor.py` + commande CLI `doctor`
  - `agent/pr_ready.py` inclut quality gate + checklist validation dans PR markdown

---

## KPI cibles a 30 jours

- [ ] Taux succes `run --apply` sans rollback >= 70%
- [ ] Echec JSON/planner < 5%
- [ ] Temps moyen d'une tache < 3 min
- [ ] PR-ready utilisable sans retouche majeure >= 60%

## KPI actuels

- Mesure date: 2026-02-18T00:24:21.650322
- Success rate: 100.0% (sample eval: 8/8)
- JSON failure rate: 32.0%
- Temps moyen: 8.76 s
- PR-ready usability: 0.0% (aucun event pr-ready encore journalise)

---

## Definition de "Done"

Une tache est "Done" si:
- code implemente
- test manuel fait
- impact note dans ce document
- (si pertinent) couverture par eval

---

## Journal de progression

### 2026-02-17

- [x] Creation du plan 30 jours
- [x] Lancement Semaine 1
- [x] Completion Semaine 1 (schema + critique + loop guard + logs echecs)
- [x] Ajout commande `python main.py progress`
- [x] Completion Semaine 2 (AST patch + tests cibles + quality fast/full + risk score)
- [x] Completion Semaine 3 (chunking symbolique + reranking hybride + context budget + query cache)
- [x] Completion Semaine 4 (benchmark 50 + KPI auto + doctor + pr-ready checklist)
- [x] Snapshot KPI rapide via `python main.py eval --limit 8`
- Notes:
  - suivi automatisable via le markdown + commande CLI
  - eval complete 50 taches possible mais plus longue; snapshot rapide utilise pour mesure immediate

---

## Backlog bonus (apres 30 jours)

- [ ] TUI plus riche (etat agent, diff live, logs live)
- [ ] Integration IDE (commandes rapides)
- [ ] Support multi-langages plus robuste
- [ ] Generation auto de tests unitaires cibles
