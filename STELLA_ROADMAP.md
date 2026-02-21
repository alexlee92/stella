# Stella — Roadmap d'Améliorations

*Basé sur l'audit technique du 2026-02-21 — objectif : combler l'écart avec les agents de production*

**Légende :** 🔴 TODO · 🟡 EN COURS · ✅ FAIT · ❌ ANNULÉ

---

## P1 — Fondations (fiabilité critique)

| # | Amélioration | Statut | Fichier(s) | Notes |
|---|---|---|---|---|
| P1.1 | Timeout par outil | ✅ FAIT | `agent/tooling.py` | Chaque subprocess a un timeout individuel configurable |
| P1.2 | JSON schema enforcement (Ollama `format:json`) | ✅ FAIT | `agent/llm_interface.py` | Force Ollama à retourner du JSON valide nativement |
| P1.3 | Parser les tracebacks en objets structurés | ✅ FAIT | `agent/traceback_parser.py` (nouveau) | Fichier, ligne, type d'erreur → dict structuré |

---

## P2 — Contexte (qualité des décisions)

| # | Amélioration | Statut | Fichier(s) | Notes |
|---|---|---|---|---|
| P2.1 | Context adaptatif (budget dynamique) | ✅ FAIT | `agent/memory.py` | Budget ajusté selon complexité du goal |
| P2.2 | Cross-file dependency tracking | ✅ FAIT | `agent/dependency_graph.py` (nouveau) | Graphe d'imports Python pour inclure les fichiers liés |
| P2.3 | Reranking MMR des chunks mémoire | ✅ FAIT | `agent/memory.py` | Maximal Marginal Relevance — diversifie les résultats |

---

## P3 — Performance (vitesse)

| # | Amélioration | Statut | Fichier(s) | Notes |
|---|---|---|---|---|
| P3.1 | Streaming des réponses LLM | ✅ FAIT | `agent/llm_interface.py` | `ask_llm_stream()` avec affichage progressif |
| P3.2 | Parallélisation des outils indépendants | ✅ FAIT | `agent/tooling.py` | `read_many` parallélisé via ThreadPoolExecutor |
| P3.3 | Cache de résultats d'outils | ✅ FAIT | `agent/tooling.py` | LRU cache TTL pour `read_file` et `list_files` |

---

## P4 — Multi-langage & Qualité

| # | Amélioration | Statut | Fichier(s) | Notes |
|---|---|---|---|---|
| P4.1 | AST merge JS/TS via tree-sitter | ✅ FAIT | `agent/patcher.py`, `agent/ts_merge.py` (nouveau) | Merge symbol-aware pour JS/TS/JSX/TSX |
| P4.2 | mypy / pyright intégration | ✅ FAIT | `agent/quality.py` | Étape optionnelle de type-checking |
| P4.3 | Coverage-guided test generation | ✅ FAIT | `agent/test_generator.py` | pytest-cov pour guider la génération |

---

## Résumé de progression

```
P1 Fondations    : [███████████] 3/3  ✅
P2 Contexte      : [███████████] 3/3  ✅
P3 Performance   : [███████████] 3/3  ✅
P4 Qualité       : [███████████] 3/3  ✅

Total            : [███████████] 12/12 ✅
```

---

## Score avant/après

| Dimension | Avant | Après | Delta |
|---|---|---|---|
| Fiabilité JSON | ~87% parallèle | ~99% (format natif) | +12% |
| Contexte moyen | 1 700 tokens fixes | 1 700–12 000 adaptatif | +6x max |
| Vitesse read_many | séquentiel | parallèle | ~3x |
| Langages AST merge | Python seulement | Python + JS/TS/JSX/TSX | +4 |
| Qualité type-check | aucun | mypy optionnel | nouveau |
| Diversité contexte | sans reranking | MMR | meilleur recall |
| Feedback utilisateur | silencieux (93s) | streaming progressif | UX majeur |
| Score global | 5/10 | 7.5/10 | +2.5 |

---

*Fichier maintenu manuellement — mettre à jour le statut après chaque implémentation*
