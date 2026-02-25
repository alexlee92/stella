import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse

import signal
from datetime import datetime

from agent.agent import (
    apply_suggestion,
    ask_project_stream,
    index_project,
    patch_risk,
    propose_file_update,
    review_file_update,
)
from agent.auto_agent import AutonomousAgent
from agent.bootstrap import format_bootstrap, run_bootstrap
from agent.chat_session import ChatSession
from agent.config import AUTO_MAX_STEPS
from agent.dev_task import ide_shortcuts, run_dev_task
from agent.doctor import format_doctor, run_doctor
from agent.eval_runner import run_eval
from agent.git_tools import (
    changed_files,
    current_branch,
    git_log,
    git_stash,
    git_stash_pop,
)
from agent.patcher import find_latest_backup, restore_backup
from agent.pr_ready import prepare_pr
from agent.progress import summarize_progress
from agent.project_map import render_project_map


def _smart_dispatch(goal: str) -> None:
    """Route automatiquement vers ask / run / fix selon l'intention détectée."""
    index_project()
    low = goal.strip().lower()

    # --- Question (lecture seule, réponse rapide) ---
    question_starters = (
        "qu'est",
        "qu'",
        "c'est quoi",
        "c'est quoi",
        "qu est",
        "quel",
        "quelle",
        "comment",
        "pourquoi",
        "explique",
        "expliques",
        "dis moi",
        "dis-moi",
        "what ",
        "how ",
        "why ",
        "explain",
        "is there",
        "does ",
        "where ",
        "where is",
        "show me",
        "liste ",
        "listes ",
    )
    is_question = goal.strip().endswith("?") or any(
        low.startswith(s) for s in question_starters
    )

    # --- Création de nouveaux fichiers (agent autonome multi-étapes) ---
    create_keywords = (
        "crée ",
        "génère ",
        "implémente ",
        "implémente un",
        "ajoute un module",
        "creer ",
        "créer ",
        "genere ",
        "générer ",
        "implemente ",
        "create ",
        "generate ",
        "scaffold",
        "nouveau module",
        "new module",
        "nouveau fichier",
        "new file",
    )
    is_creation = any(kw in low for kw in create_keywords)

    if is_question:
        print("[stella] mode detecte : question -- reponse directe\n")
        ask_project_stream(goal)
    elif is_creation:
        print("[stella] mode détecté : création — agent autonome\n")
        print(AutonomousAgent(max_steps=10).run(goal=goal))
    else:
        print("[stella] mode détecté : modification — fix standard\n")
        result = run_dev_task(goal=goal, profile="standard")
        status = result.get("status", "?")
        changed = result.get("changed_files_count", 0)
        next_action = result.get("next_action", "")
        print(f"\nStatut    : {status}")
        print(f"Fichiers  : {changed} modifié(s)")
        if next_action:
            print(f"Prochaine étape : {next_action}")
        summary_md = result.get("summary_md")
        if summary_md:
            print(f"Rapport complet : {summary_md}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Local coding agent (DeepSeek/Ollama)",
        usage='stella.py [command] [goal]  — ou simplement : stella.py "<ton goal>"',
    )
    sub = parser.add_subparsers(dest="command", required=False)

    idx = sub.add_parser(
        "index", help="Index project files into persistent vector memory"
    )
    idx.add_argument(
        "--rebuild", action="store_true", help="Force rebuild memory index"
    )

    ask = sub.add_parser("ask", help="Ask a question about the codebase")
    ask.add_argument("question", help="Question to ask")

    review = sub.add_parser("review", help="Generate patch and show diff only")
    review.add_argument("file", help="Target file path")
    review.add_argument("instruction", help="Change request")

    apply_cmd = sub.add_parser("apply", help="Generate patch and apply immediately")
    apply_cmd.add_argument("file", help="Target file path")
    apply_cmd.add_argument("instruction", help="Change request")
    apply_cmd.add_argument(
        "--non-interactive", action="store_true", help="Apply without confirmation"
    )
    apply_cmd.add_argument(
        "--with-tests",
        action="store_true",
        help="Generate targeted unit tests after applying code changes",
    )

    undo = sub.add_parser("undo", help="Restore latest backup for file")
    undo.add_argument("file", help="File to restore")

    plan = sub.add_parser("plan", help="Ask planner for next action only")
    plan.add_argument("goal", help="Goal for the autonomous agent")

    run = sub.add_parser("run", help="Run autonomous tool-use loop")
    run.add_argument("goal", help="Goal for the autonomous agent")
    run.add_argument(
        "--steps", type=int, default=AUTO_MAX_STEPS, help="Maximum decision steps"
    )
    run.add_argument("--apply", action="store_true", help="Allow applying staged edits")
    run.add_argument(
        "--fix-until-green",
        action="store_true",
        help="Enable deterministic replanning loop until checks are green (bounded)",
    )
    run.add_argument(
        "--with-tests",
        action="store_true",
        help="Generate targeted unit tests for modified Python files",
    )
    run.add_argument(
        "--max-seconds",
        type=int,
        default=0,
        help="Hard runtime budget in seconds (0 = disabled)",
    )

    pr = sub.add_parser(
        "pr-ready", help="Create branch, commit changes, and print PR summary"
    )
    pr.add_argument("goal", help="Goal used for default branch/commit naming")
    pr.add_argument("--branch", help="Optional branch name")
    pr.add_argument("--message", help="Optional commit message")

    chat = sub.add_parser("chat", help="Start continuous chat with session memory")
    chat.add_argument(
        "--steps", type=int, default=AUTO_MAX_STEPS, help="Default max steps for /run"
    )
    chat.add_argument(
        "--apply", action="store_true", help="Allow applying staged edits in /run"
    )
    chat.add_argument(
        "--fix-until-green",
        action="store_true",
        help="Enable deterministic replanning loop until checks are green (bounded)",
    )
    chat.add_argument(
        "--with-tests",
        action="store_true",
        help="Generate targeted unit tests for modified Python files",
    )
    chat.add_argument("--max-seconds", type=int, default=0)

    bootstrap = sub.add_parser(
        "bootstrap", help="Auto-setup git/index/tools for local use"
    )
    bootstrap.add_argument(
        "--no-git-init", action="store_true", help="Skip git init step"
    )
    bootstrap.add_argument(
        "--no-index-rebuild", action="store_true", help="Skip memory index rebuild"
    )
    bootstrap.add_argument(
        "--no-install-tools",
        action="store_true",
        help="Skip pip install pytest/ruff/black",
    )

    sub.add_parser("map", help="Print project symbol map")
    eval_cmd = sub.add_parser("eval", help="Run evaluation suite from eval/tasks.json")
    eval_cmd.add_argument(
        "--tasks",
        default="eval/tasks.json",
        help="Tasks file path relative to project root (e.g. eval/tasks_code_edit.json)",
    )
    eval_cmd.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only first N tasks for quick KPI snapshot",
    )
    sub.add_parser("doctor", help="Run environment and tooling diagnostics")
    sub.add_parser("ci", help="Run local compile checks")
    sub.add_parser("progress", help="Show progress summary from UPGRADE_PLAN_30J.md")
    dev_task = sub.add_parser(
        "dev-task", help="Single command: plan + patch + tests + actionable summary"
    )
    dev_task.add_argument("goal", help="Goal for the autonomous agent")
    dev_task.add_argument("--steps", type=int, default=AUTO_MAX_STEPS)
    dev_task.add_argument("--apply", action="store_true")
    dev_task.add_argument("--fix-until-green", action="store_true")
    dev_task.add_argument("--with-tests", action="store_true")
    dev_task.add_argument("--profile", choices=["safe", "standard", "aggressive"])
    dev_task.add_argument("--max-seconds", type=int, default=0)
    sub.add_parser("ide-shortcuts", help="Print quick IDE-friendly command shortcuts")

    auto = sub.add_parser("auto", help="Alias of run")
    auto.add_argument("goal", help="Goal for the autonomous agent")
    auto.add_argument("--steps", type=int, default=AUTO_MAX_STEPS)
    auto.add_argument("--apply", action="store_true")
    auto.add_argument("--fix-until-green", action="store_true")
    auto.add_argument("--with-tests", action="store_true")
    auto.add_argument("--max-seconds", type=int, default=0)

    edit = sub.add_parser("edit", help="Alias of review/apply workflow")
    edit.add_argument("file")
    edit.add_argument("instruction")
    edit.add_argument("--apply", action="store_true")

    # Commandes simplifiées pour non-experts
    init_cmd = sub.add_parser(
        "init",
        help="Initialiser Stella sur ce projet (git + index + outils) — pour débutants",
    )
    init_cmd.add_argument(
        "--no-git", action="store_true", help="Ne pas initialiser git"
    )

    test_cmd = sub.add_parser(
        "test",
        help="Lancer les tests rapidement (alias de pytest -q)",
    )
    test_cmd.add_argument(
        "args", nargs="*", default=[], help="Arguments pytest optionnels"
    )

    scaffold_cmd = sub.add_parser(
        "scaffold",
        help="Generer un fichier a partir d'un template (fastapi-endpoint, django-model, react-component, ...)",
    )
    scaffold_cmd.add_argument("type", help="Type de template")
    scaffold_cmd.add_argument("name", help="Nom de l'entite")
    scaffold_cmd.add_argument("--output-dir", default="", help="Dossier de sortie")

    watch_cmd = sub.add_parser(
        "watch",
        help="Surveiller les fichiers et relancer les tests automatiquement",
    )
    watch_cmd.add_argument(
        "--pattern", default="**/*.py", help="Glob pattern a surveiller"
    )
    watch_cmd.add_argument(
        "--command",
        dest="watch_command",
        default="",
        help="Commande a lancer (defaut: python -m pytest -q)",
    )
    watch_cmd.add_argument(
        "--interval", type=float, default=2.0, help="Intervalle de scan en secondes"
    )

    fix_cmd = sub.add_parser(
        "fix",
        help="Corriger / améliorer le code en langage naturel (mode standard, apply auto)",
    )
    fix_cmd.add_argument("description", help="Ce que tu veux faire ou corriger")
    fix_cmd.add_argument(
        "--safe", action="store_true", help="Mode prudent : propose sans appliquer"
    )
    fix_cmd.add_argument(
        "--aggressive",
        action="store_true",
        help="Mode agressif : correction itérative jusqu'aux tests verts",
    )

    return parser


_CHAT_HELP = """
Commandes disponibles :
  /run <objectif>     -- Lancer l'agent autonome sur un objectif
  /plan <objectif>    -- Afficher le plan sans l'executer
  /ask <question>     -- Poser une question sur le codebase (streaming)
  /status             -- Afficher l'etat git (branche, fichiers modifies)
  /map                -- Afficher la carte des symboles du projet
  /log [fichier]      -- Historique des commits recents
  /stash [message]    -- Sauvegarder le travail en cours
  /stash-pop          -- Restaurer le dernier stash
  /undo <fichier>     -- Annuler la derniere modification d'un fichier
  /eval               -- Lancer les tests rapides (pytest -q)
  /decisions          -- Afficher les dernieres decisions de l'agent
  /context            -- Voir ce que l'agent sait (memoire de session)
  /goto <fichier> <symbole> -- Trouver la definition d'un symbole
  /refs <fichier> <symbole> -- Trouver les references d'un symbole
  /symbols <fichier>  -- Lister les symboles d'un fichier
  /sessions           -- Lister les sessions precedentes
  /replay [id]        -- Reprendre une session precedente
  /help               -- Afficher cette aide
  /exit               -- Quitter le chat
"""


_CHAT_COMMANDS = [
    "/run",
    "/plan",
    "/ask",
    "/status",
    "/map",
    "/log",
    "/stash",
    "/stash-pop",
    "/undo",
    "/eval",
    "/decisions",
    "/context",
    "/goto",
    "/refs",
    "/symbols",
    "/sessions",
    "/replay",
    "/test",
    "/scaffold",
    "/help",
    "/exit",
]


def _build_prompt_session():
    """P4.2 — Build a prompt_toolkit session with auto-completion and history."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import InMemoryHistory

        completer = WordCompleter(_CHAT_COMMANDS, sentence=True)
        return PromptSession(
            completer=completer,
            history=InMemoryHistory(),
        )
    except ImportError:
        return None


def _highlight_code(text: str) -> str:
    """P4.1 — Highlight code blocks in agent output using rich if available."""
    try:
        from rich.console import Console
        from rich.syntax import Syntax
        import io
        import re

        # Find ```lang ... ``` blocks and colorize them
        pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)

        def _replace(m):
            lang = m.group(1) or "python"
            code = m.group(2).strip()
            buf = io.StringIO()
            console = Console(file=buf, force_terminal=True, width=100)
            console.print(Syntax(code, lang, theme="monokai", line_numbers=False))
            return buf.getvalue().strip()

        return pattern.sub(_replace, text)
    except Exception:
        return text


def run_chat(
    default_steps: int,
    auto_apply: bool,
    fix_until_green: bool,
    with_tests: bool,
    max_seconds: int,
):
    index_project()
    session = ChatSession()
    prompt_session = _build_prompt_session()

    print(_CHAT_HELP)
    while True:
        try:
            if prompt_session:
                user_input = prompt_session.prompt("you> ").strip()
            else:
                user_input = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended")
            break

        if not user_input:
            continue

        if user_input in {"/exit", "exit", "quit"}:
            print("Session ended")
            break

        if user_input == "/help":
            print(_CHAT_HELP)
            continue

        if user_input.startswith("/run "):
            goal = user_input[len("/run ") :].strip()
            if not goal:
                print("[!] Usage : /run <objectif>")
                continue
            summary = session.run_auto(
                goal=goal,
                auto_apply=auto_apply,
                max_steps=default_steps,
                fix_until_green=fix_until_green,
                generate_tests=with_tests,
                max_seconds=max_seconds,
            )
            print(summary)
            continue

        if user_input.startswith("/plan "):
            goal = user_input[len("/plan ") :].strip()
            if not goal:
                print("[!] Usage : /plan <objectif>")
                continue
            decision = AutonomousAgent(max_steps=1).plan_once(goal)
            print(decision)
            continue

        if user_input.startswith("/ask "):
            question = user_input[len("/ask ") :].strip()
            if not question:
                print("[!] Usage : /ask <question>")
                continue
            ask_project_stream(question)
            continue

        if user_input == "/status":
            branch = current_branch() or "inconnue"
            changed = changed_files()
            print("\n--- Status du projet ---")
            print(f"  Branche : {branch}")
            print(f"  Fichiers modifies : {len(changed)}")
            if changed:
                for f in changed[:20]:
                    ext = os.path.splitext(f)[1]
                    print(f"    {f} ({ext})")
                if len(changed) > 20:
                    print(f"    ... et {len(changed) - 20} autres")
            print(
                f"  Session chat : {len(session.messages)} messages, {len(session.decisions)} decisions"
            )
            print()
            continue

        if user_input == "/map":
            print(render_project_map())
            continue

        if user_input.startswith("/log"):
            file_arg = user_input[len("/log") :].strip() or None
            print(git_log(file_path=file_arg))
            continue

        if user_input.startswith("/stash-pop"):
            code, out = git_stash_pop()
            print(out)
            continue

        if user_input.startswith("/stash"):
            msg = user_input[len("/stash") :].strip() or None
            code, out = git_stash(message=msg)
            print(out)
            continue

        if user_input.startswith("/undo "):
            filepath = user_input[len("/undo ") :].strip()
            if not filepath:
                print("[!] Usage : /undo <fichier>")
                continue
            backup = find_latest_backup(filepath)
            if not backup:
                print(f"[!] Aucun backup trouvé pour {filepath}")
            else:
                ok = restore_backup(filepath, backup)
                print("Annulé avec succès." if ok else "[!] Échec de l'annulation.")
            continue

        if user_input == "/eval":
            from agent.tooling import run_tests

            print("Lancement des tests...")
            result = run_tests("pytest -q")
            print(result)
            continue

        if user_input == "/decisions":
            print(session.show_decisions())
            continue

        if user_input == "/context":
            print(session.show_context())
            continue

        if user_input.startswith("/goto "):
            from agent.code_intelligence import goto_definition

            parts = user_input[len("/goto ") :].strip().split()
            if len(parts) < 2:
                print("  Usage: /goto <fichier> <symbole>")
            else:
                results = goto_definition(parts[0], parts[1])
                for r in results:
                    if "error" in r:
                        print(f"  {r['error']}")
                    else:
                        print(
                            f"  {r['file']}:{r['line']}:{r['column']} [{r['type']}] {r['name']}"
                        )
            continue

        if user_input.startswith("/refs "):
            from agent.code_intelligence import find_references

            parts = user_input[len("/refs ") :].strip().split()
            if len(parts) < 2:
                print("  Usage: /refs <fichier> <symbole>")
            else:
                results = find_references(parts[0], parts[1])
                for r in results:
                    if "error" in r:
                        print(f"  {r['error']}")
                    else:
                        print(f"  {r['file']}:{r['line']} {r['context']}")
            continue

        if user_input.startswith("/symbols "):
            from agent.code_intelligence import list_symbols

            file_arg = user_input[len("/symbols ") :].strip()
            if not file_arg:
                print("  Usage: /symbols <fichier>")
            else:
                syms = list_symbols(file_arg)
                if not syms:
                    print("  Aucun symbole trouve.")
                for s in syms:
                    print(f"  L{s['line']} [{s['type']}] {s['name']}")
            continue

        # P2.2 — Session persistence commands
        if user_input == "/sessions":
            from agent.chat_session import list_sessions

            sessions_list = list_sessions()
            if not sessions_list:
                print("  Aucune session sauvegardee.")
            else:
                print("\n--- Sessions precedentes ---")
                for s in sessions_list:
                    ts = (
                        datetime.fromtimestamp(s["mtime"]).strftime("%Y-%m-%d %H:%M")
                        if s["mtime"]
                        else "?"
                    )
                    goal = s.get("goal", "")[:60]
                    print(f"  {s['id']}  {ts}  {goal}")
                print()
            continue

        if user_input.startswith("/replay"):
            from agent.chat_session import get_latest_session_id

            sid = user_input[len("/replay") :].strip()
            if not sid:
                sid = get_latest_session_id()
                if not sid:
                    print("  Aucune session a reprendre.")
                    continue
            ok = session.load_session(sid)
            if ok:
                print(
                    f"  Session {sid} restauree : {len(session.messages)} messages, {len(session.staged_edits)} edits en attente."
                )
                if session.staged_edits:
                    print(
                        f"  Fichiers staged : {', '.join(session.staged_edits.keys())}"
                    )
            else:
                print(f"  Session {sid} introuvable.")
            continue

        # P4.4 — /test alias for quick test run
        if user_input == "/test" or user_input.startswith("/test "):
            from agent.tooling import run_tests

            test_args = user_input[len("/test") :].strip()
            cmd = f"pytest -q {test_args}".strip()
            print(f"Lancement : {cmd}")
            print(run_tests(cmd))
            continue

        # P3.7 — /scaffold command
        if user_input.startswith("/scaffold "):
            from agent.scaffolder import scaffold

            parts = user_input[len("/scaffold ") :].strip().split(None, 1)
            if len(parts) < 2:
                print("  Usage: /scaffold <type> <name>")
                print(
                    "  Types: fastapi-endpoint, django-model, django-view, react-component, python-module, test"
                )
            else:
                result = scaffold(parts[0], parts[1])
                print(result)
            continue

        answer = session.ask(user_input)
        print(_highlight_code(answer))


def handle_edit_like(file_path: str, instruction: str, do_apply: bool):
    index_project()
    code = propose_file_update(file_path, instruction)
    diff = review_file_update(file_path, code)
    risk = patch_risk(file_path, code)

    print(
        f"[risk] level={risk['level']} score={risk['score']} changed_lines={risk['changed_lines']} sensitive={risk['sensitive_hits']}"
    )
    print(diff or "(no diff)")

    if do_apply:
        apply_suggestion(file_path, code, interactive=True)


_KNOWN_COMMANDS = {
    "index",
    "ask",
    "review",
    "apply",
    "undo",
    "plan",
    "run",
    "pr-ready",
    "chat",
    "bootstrap",
    "map",
    "eval",
    "doctor",
    "ci",
    "progress",
    "dev-task",
    "ide-shortcuts",
    "auto",
    "edit",
    "init",
    "fix",
    "test",
    "scaffold",
    "watch",
}


def _handle_sigint(signum, frame):
    """P1.5 — Gestion gracieuse de Ctrl+C."""
    print("\n\n[stella] Interruption detectee (Ctrl+C). Arret en cours...")
    print("[stella] Les fichiers non appliques sont preserves dans staged_edits.")
    raise SystemExit(130)


def main():
    # P1.5 — Gestion gracieuse de Ctrl+C
    signal.signal(signal.SIGINT, _handle_sigint)

    # Entrée directe : python stella.py "mon goal" sans sous-commande
    # Si le premier argument n'est pas une sous-commande connue → dispatch auto
    if (
        len(sys.argv) >= 2
        and sys.argv[1] not in _KNOWN_COMMANDS
        and not sys.argv[1].startswith("-")
    ):
        goal = " ".join(sys.argv[1:])
        _smart_dispatch(goal)
        return

    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "index":
        index_project(force_rebuild=args.rebuild)
        return

    if args.command == "ask":
        index_project()
        ask_project_stream(args.question)
        return

    if args.command == "review":
        handle_edit_like(args.file, args.instruction, do_apply=False)
        return

    if args.command == "apply":
        index_project()
        code = propose_file_update(args.file, args.instruction)
        risk = patch_risk(args.file, code)
        print(
            f"[risk] level={risk['level']} score={risk['score']} changed_lines={risk['changed_lines']} sensitive={risk['sensitive_hits']}"
        )
        if args.non_interactive:
            print(apply_suggestion(args.file, code, interactive=False))
        else:
            apply_suggestion(args.file, code, interactive=True)
        if args.with_tests:
            from agent.test_generator import apply_generated_tests

            generated = apply_generated_tests([args.file], limit=2)
            print(
                f"[tests] generated={generated.get('generated', [])} applied={generated.get('applied', [])}"
            )
        return

    if args.command == "undo":
        backup = find_latest_backup(args.file)
        if not backup:
            print("No backup found")
            return
        ok = restore_backup(args.file, backup)
        print("undo ok" if ok else "undo failed")
        return

    if args.command == "plan":
        index_project()
        print(AutonomousAgent(max_steps=1).plan_once(args.goal))
        return

    if args.command in {"run", "auto"}:
        index_project()
        print(
            AutonomousAgent(max_steps=args.steps, interactive=True).run(
                goal=args.goal,
                auto_apply=args.apply,
                fix_until_green=args.fix_until_green,
                generate_tests=args.with_tests,
                max_seconds=args.max_seconds,
            )
        )
        return

    if args.command == "pr-ready":
        result = prepare_pr(
            goal=args.goal, branch=args.branch, commit_message=args.message
        )
        print(result["summary"])
        if not result.get("ok"):
            raise SystemExit(1)
        return

    if args.command == "chat":
        run_chat(
            default_steps=args.steps,
            auto_apply=args.apply,
            fix_until_green=args.fix_until_green,
            with_tests=args.with_tests,
            max_seconds=args.max_seconds,
        )
        return

    if args.command == "bootstrap":
        report = run_bootstrap(
            init_git=not args.no_git_init,
            rebuild_index=not args.no_index_rebuild,
            install_tools=not args.no_install_tools,
        )
        print(format_bootstrap(report))
        if not report.get("ok"):
            raise SystemExit(1)
        return

    if args.command == "map":
        print(render_project_map())
        return

    if args.command == "eval":
        report = run_eval(
            tasks_file=args.tasks, max_tasks=args.limit if args.limit > 0 else None
        )
        print(report["summary"])
        print(report.get("kpis", {}))
        return

    if args.command == "doctor":
        result = run_doctor()
        print(format_doctor(result))
        if result.get("failed", 0) > 0:
            raise SystemExit(1)
        return

    if args.command == "ci":
        import scripts_run_ci

        scripts_run_ci.main()
        return

    if args.command == "progress":
        print(summarize_progress())
        return

    if args.command == "dev-task":
        result = run_dev_task(
            goal=args.goal,
            max_steps=args.steps,
            auto_apply=args.apply,
            fix_until_green=args.fix_until_green,
            with_tests=args.with_tests,
            profile=args.profile,
            max_seconds=args.max_seconds,
        )
        print(
            {
                "status": result.get("status"),
                "changed_files_count": result.get("changed_files_count"),
                "next_action": result.get("next_action"),
                "summary_json": result.get("summary_json"),
                "summary_md": result.get("summary_md"),
            }
        )
        return

    if args.command == "ide-shortcuts":
        print(ide_shortcuts())
        return

    if args.command == "edit":
        handle_edit_like(args.file, args.instruction, do_apply=args.apply)
        return

    if args.command == "init":
        print("=== Stella — Initialisation du projet ===\n")
        report = run_bootstrap(
            init_git=not args.no_git,
            rebuild_index=True,
            install_tools=True,
        )
        print(format_bootstrap(report))
        if report.get("ok"):
            print("\nProchaines étapes :")
            print('  python stella.py fix "<décris ce que tu veux faire>"')
            print("  python stella.py chat          — mode interactif")
            print("  python stella.py doctor        — diagnostics environnement")
        else:
            raise SystemExit(1)
        return

    if args.command == "fix":
        profile = (
            "safe" if args.safe else ("aggressive" if args.aggressive else "standard")
        )
        print(f"=== Stella Fix — profil : {profile} ===")
        print(f"Objectif : {args.description}\n")
        result = run_dev_task(
            goal=args.description,
            profile=profile,
        )
        status = result.get("status", "?")
        changed = result.get("changed_files_count", 0)
        next_action = result.get("next_action", "")
        print(f"\nStatut    : {status}")
        print(f"Fichiers  : {changed} modifié(s)")
        print(f"Prochaine étape : {next_action}")
        summary_md = result.get("summary_md")
        if summary_md:
            print(f"Rapport complet : {summary_md}")
        return

    # P4.4 — stella test (alias rapide pour pytest)
    if args.command == "test":
        from agent.tooling import run_tests

        extra = " ".join(args.args) if args.args else ""
        cmd = f"python -m pytest -q {extra}".strip()
        print("=== Stella Test ===")
        print(f"Commande : {cmd}\n")
        print(run_tests(cmd))
        return

    # P3.7 — stella scaffold
    if args.command == "scaffold":
        from agent.scaffolder import scaffold as do_scaffold

        result = do_scaffold(args.type, args.name, output_dir=args.output_dir)
        print(result)
        return

    # P4.5 — stella watch
    if args.command == "watch":
        from agent.watcher import run_watch

        run_watch(
            pattern=args.pattern,
            command=args.watch_command if args.watch_command else None,
            interval=args.interval,
        )
        return


if __name__ == "__main__":
    main()
