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
        [sys.executable, "-m", "py_compile", "main.py"],
        [
            sys.executable,
            "-m",
            "py_compile",
            "agent\\agent.py",
            "agent\\auto_agent.py",
            "agent\\chat_session.py",
        ],
    ]

    for cmd in steps:
        code = run(cmd)
        if code != 0:
            raise SystemExit(code)

    print("ci local passed")


if __name__ == "__main__":
    main()
