import subprocess
import sys


def run(command: list[str]):
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    print(f"$ {' '.join(command)}")
    print(out.strip() or "(no output)")
    return proc.returncode


def main():
    steps = [
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "pytest", "-q"],
    ]

    for cmd in steps:
        code = run(cmd)
        if code != 0:
            raise SystemExit(code)

    print("ci local passed")


if __name__ == "__main__":
    main()
