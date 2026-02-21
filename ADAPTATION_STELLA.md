# Plan d'Adaptation de Stella aux Modèles Orisha

Ce document suit l'évolution de l'intégration des modèles `Orisha-Ifa1.0` et `Orisha-Oba1.0` dans l'agent Stella via le proxy Flask.

## 📋 Tâches à accomplir

- [x] **Phase 1 : Analyse de l'existant**
    - [x] Identifier la classe de base pour les clients LLM dans `agent/llm.py` ou `agent/llm_interface.py`.
    - [x] Analyser la gestion de la configuration dans `agent/config.py` ou `agent/settings.py`.
- [x] **Phase 2 : Développement du Connecteur Orisha**
    - [x] Créer un nouveau client LLM capable de requêter l'API Flask (port 5000).
    - [x] Implémenter la logique de mapping des tâches (`task_type`) vers les modèles Orisha.
- [x] **Phase 3 : Intégration et Configuration**
    - [x] Ajouter les options de configuration pour l'URL de l'API Flask dans `settings.toml` ou `.env`.
    - [x] Modifier la factory de modèles pour instancier `OrishaClient` quand spécifié.
- [x] **Phase 4 : Tests et Optimisation**
    - [x] Vérifier la bonne transmission des prompts et la réception des réponses.
    - [x] Tester le routage intelligent selon le type de tâche (Analyse vs Développement).
    - [x] Valider la gestion des contextes (num_ctx).

## 🚀 État d'avancement
- **Dernière mise à jour :** 21/02/2026
- **Statut actuel :** Terminé. Stella utilise maintenant les modèles Orisha via le proxy Flask.
