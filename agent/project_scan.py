import os
from agent.config import PROJECT_ROOT, MAX_FILE_SIZE

EXCLUDED_DIRS = {
    ".venv",
    "__pycache__",
    ".git",
    ".idea",
    "site-packages",
    "build",
    "dist",
}


def get_python_files(project_root=None):
    root_dir = os.path.abspath(project_root or PROJECT_ROOT)
    files = []

    for root, dirs, filenames in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            path = os.path.join(root, filename)
            try:
                if os.path.getsize(path) <= MAX_FILE_SIZE:
                    files.append(path)
            except OSError:
                continue

    return files


def load_file_content(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return ""
