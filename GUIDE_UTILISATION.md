# Guide d'Utilisation : Stella Agent

Ce guide explique comment utiliser Stella Agent comme assistant de programmation pour vos projets. Stella est optimisé pour fonctionner localement avec Ollama ou via l'API OpenAI.

---

## 1. Installation et Configuration

### Prérequis
- **Python 3.11+**
- **Ollama** (pour l'usage local) : [ollama.com](https://ollama.com)
  - Modèle recommandé : `ollama pull deepseek-coder:6.7b`
  - Modèle d'embedding : `ollama pull nomic-embed-text`

### Installation
Clonez Stella dans un dossier, puis dans votre projet cible (ou dans le dossier de Stella pour travailler sur Stella lui-même) :

```bash
# Créer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Ou .venv\Scripts\activate sur Windows

# Installer les dépendances (maintenant incluant tqdm, ruff, black, etc.)
pip install -e .
```

### Configuration (`settings.toml`)
Configurez vos préférences dans le fichier `settings.toml` à la racine :

```toml
[models]
main = "deepseek-coder:6.7b" # Ou "gpt-4" si vous avez une clé
embed = "nomic-embed-text"

[ollama]
base_url = "http://localhost:11434"

[agent]
# Pour utiliser OpenAI (optionnel)
# OPENAI_API_KEY = "sk-..." 
```

---

## 2. Étape 1 : Indexer votre projet
Avant de poser des questions complexes, Stella doit "apprendre" votre code. 

```bash
python main.py index
```
*Note : Grâce à la mise à jour, une barre de progression s'affiche et l'indexation est parallélisée.*

---

## 3. Étape 2 : Mode Interactif (Quick Tasks)

### Poser une question sur le code
Stella utilise sa mémoire sémantique pour trouver les fichiers pertinents.
```bash
python main.py ask "Comment est gérée l'authentification dans ce projet ?"
```

### Proposer une modification (Review)
Pour voir ce que Stella changerait sans appliquer la modification :
```bash
python main.py review chemin/vers/fichier.py "Ajoute une validation pour l'email"
```

### Appliquer une modification
Stella va maintenant privilégier les blocs **SEARCH/REPLACE** pour ne modifier que les lignes nécessaires sans reformatage global.
```bash
python main.py apply chemin/vers/fichier.py "Refactorise la boucle pour utiliser une liste en compréhension"
```

---

## 4. Étape 3 : Mode Autonome (Agentic Mode)

C'est ici que Stella devient un véritable agent. Il peut lire plusieurs fichiers, chercher dans le code et exécuter des tests jusqu'à ce que la solution soit trouvée.

### Résoudre un bug ou implémenter une feature
```bash
python main.py run "Fixe le bug de timeout dans le module de téléchargement" --apply --steps 10
```

### Mode Robuste (Auto-correction)
Stella va essayer de corriger le code jusqu'à ce que les tests passent :
```bash
python main.py run "Crée un nouveau endpoint API pour les rapports" --apply --with-tests --fix-until-green
```

---

## 5. Étape 4 : Mode Chat Continu

Pour une expérience proche de ChatGPT/Claude mais avec accès à vos fichiers :

```bash
python main.py chat --apply
```
**Commandes dans le chat :**
- `/plan <but>` : Demander à l'agent de réfléchir à une stratégie.
- `/run <but>` : Lancer l'exécution autonome.
- `Votre message` : Chat normal avec contexte du code.

---

## 6. Bonnes Pratiques pour Stella

1.  **Soyez spécifique** : Au lieu de "Améliore le code", dites "Améliore la gestion des erreurs dans `auth.py` en ajoutant des blocs try/except".
2.  **Utilisez le Quality Pipeline** : Laissez Stella exécuter ses tests. S'il fait une erreur, il fera un **rollback** automatique pour ne pas laisser votre projet dans un état instable.
3.  **Vérifiez les Diff** : Même si Stella est intelligent, vérifiez toujours les modifications appliquées (via `git diff`) avant de commiter.
4.  **Indexation régulière** : Si vous modifiez beaucoup de fichiers manuellement, relancez `python main.py index` pour mettre à jour sa mémoire.

---

## 7. Exemples de commandes courantes

| Besoin | Commande |
| :--- | :--- |
| **Expliquer un bug** | `python main.py ask "Pourquoi j'ai une erreur NullPointer ici ?"` |
| **Générer des tests** | `python main.py apply mon_script.py "Génère des tests pytest exhaustifs"` |
| **Nettoyer le code** | `python main.py run "Passe ruff sur tout le projet et fixe les erreurs" --apply` |
| **Préparer une PR** | `python main.py pr-ready "Ajout du module de logs"` |
