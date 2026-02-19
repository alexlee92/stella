import argparse

from agent.agent import (
    apply_suggestion,
    ask_project,
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
from agent.patcher import find_latest_backup, restore_backup
from agent.pr_ready import prepare_pr
from agent.progress import summarize_progress
from agent.project_map import render_project_map


def build_parser():
    parser = argparse.ArgumentParser(description="Local coding agent (DeepSeek/Ollama)")
    sub = parser.add_subparsers(dest="command", required=True)

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

    return parser


def run_chat(
    default_steps: int,
    auto_apply: bool,
    fix_until_green: bool,
    with_tests: bool,
    max_seconds: int,
):
    index_project()
    session = ChatSession()

    print("Chat mode started. Commands: /run <goal>, /plan <goal>, /decisions, /exit")
    while True:
        user_input = input("you> ").strip()
        if not user_input:
            continue

        if user_input in {"/exit", "exit", "quit"}:
            print("Session ended")
            break

        if user_input.startswith("/run "):
            goal = user_input[len("/run ") :].strip()
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
            decision = AutonomousAgent(max_steps=1).plan_once(goal)
            print(decision)
            continue

        if user_input == "/decisions":
            print(session.show_decisions())
            continue

        answer = session.ask(user_input)
        print(answer)


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


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        index_project(force_rebuild=args.rebuild)
        return

    if args.command == "ask":
        index_project()
        print(ask_project(args.question))
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
            AutonomousAgent(max_steps=args.steps).run(
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


if __name__ == "__main__":
    main()
