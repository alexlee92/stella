---
parent: More info
nav_order: 100
description: Stella is tightly integrated with git.
---

# Git integration

Stella works best with code that is part of a git repo.
Stella is tightly integrated with git, which makes it easy to:

  - Use the `/undo` command to instantly undo any AI changes that you don't like.
  - Go back in the git history to review the changes that stella made to your code
  - Manage a series of stella's changes on a git branch

Stella uses git in these ways:

- It asks to create a git repo if you launch it in a directory without one.
- Whenever stella edits a file, it commits those changes with a descriptive commit message. This makes it easy to undo or review stella's changes. 
- Stella takes special care before editing files that already have uncommitted changes (dirty files). Stella will first commit any preexisting changes with a descriptive commit message. 
This keeps your edits separate from stella's edits, and makes sure you never lose your work if stella makes an inappropriate change.

## In-chat commands

Stella also allows you to use 
[in-chat commands](/docs/usage/commands.html)
to perform git operations:

- `/diff` will show all the file changes since the last message you sent.
- `/undo` will undo and discard the last change.
- `/commit` to commit all dirty changes with a sensible commit message.
- `/git` will let you run raw git commands to do more complex management of your git history.

You can also manage your git history outside of stella with your preferred git tools.

## Disabling git integration

While it is not recommended, you can disable stella's use of git in a few ways:

  - `--no-auto-commits` will stop stella from git committing each of its changes.
  - `--no-dirty-commits` will stop stella from committing dirty files before applying its edits.
  - `--no-git` will completely stop stella from using git on your files. You should ensure you are keeping sensible backups of the files you are working with.
  - `--git-commit-verify` will run pre-commit hooks when making git commits. By default, stella skips pre-commit hooks by using the `--no-verify` flag (`--git-commit-verify=False`).

## Commit messages

Stella sends the `--weak-model` a copy of the diffs and the chat history
and asks it to produce a commit message.
By default, stella creates commit messages which follow
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

You can customize the
[commit prompt](https://github.com/Stella-AI/stella/blob/main/stella/prompts.py#L5)
with the `--commit-prompt` option.
You can place that on the command line, or 
[configure it via a config file or environment variables](https://stella.chat/docs/config.html).


## Commit attribution

Stella marks commits that it either authored or committed.

- If stella authored the changes in a commit, they will have "(stella)" appended to the git author and git committer name metadata.
- If stella simply committed changes (found in dirty files), the commit will have "(stella)" appended to the git committer name metadata.

You can use `--no-attribute-author` and `--no-attribute-committer` to disable
modification of the git author and committer name fields.

Additionally, you can use the following options to prefix commit messages:

- `--attribute-commit-message-author`: Prefix commit messages with 'stella: ' if stella authored the changes.
- `--attribute-commit-message-committer`: Prefix all commit messages with 'stella: ', regardless of whether stella authored the changes or not.

Finally, you can use `--attribute-co-authored-by` to have stella append a Co-authored-by trailer to the end of the commit string. 
This will disable appending `(stella)` to the git author and git committer unless you have explicitly enabled those settings.

