# Stella

Stella est un CLI de code base sur aider.

## Installation

```bash
python -m pip install -e .
```

## Utilisation

```bash
stella --help
```

Exemple avec Ollama:

```bash
stella --model ollama/qwen2.5-coder:14b-instruct-q5_K_M \
  --openai-api-base http://localhost:11434/v1 \
  --openai-api-key ollama
```

## Notes

- Entree CLI: `stella_ai.main:main`
- Le code legacy historique de `stella` a ete retire.
