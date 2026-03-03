import os
import sys
import time
from pathlib import Path

import packaging.version

import stella_ai
from stella_ai import utils
from stella_ai.dump import dump  # noqa: F401

def _version_cache_file() -> Path:
    # Prefer explicit override, then project-local cache to avoid home-dir permission issues.
    base = os.environ.get("STELLA_HOME") or os.environ.get("STELLA_HOME")
    if base:
        return Path(base) / "caches" / "versioncheck"
    return Path.cwd() / ".stella" / ".stella" / "caches" / "versioncheck"


VERSION_CHECK_FNAME = _version_cache_file()


def install_from_main_branch(io):
    """
    Install the latest version of stella from the main branch of the GitHub repository.
    """

    return utils.check_pip_install_extra(
        io,
        None,
        "Install the development version of stella from the main branch?",
        ["git+https://github.com/Stella-AI/stella.git"],
        self_update=True,
    )


def install_upgrade(io, latest_version=None):
    """
    Install the latest version of stella from PyPI.
    """

    if latest_version:
        new_ver_text = f"Newer stella version v{latest_version} is available."
    else:
        new_ver_text = "Install latest version of stella?"

    docker_image = os.environ.get("STELLA_DOCKER_IMAGE")
    if docker_image:
        text = f"""
{new_ver_text} To upgrade, run:

    docker pull {docker_image}
"""
        io.tool_warning(text)
        return True

    success = utils.check_pip_install_extra(
        io,
        None,
        new_ver_text,
        ["stella-chat"],
        self_update=True,
    )

    if success:
        io.tool_output("Re-run stella to use new version.")
        sys.exit()

    return


def check_version(io, just_check=False, verbose=False):
    if not just_check and VERSION_CHECK_FNAME.exists():
        day = 60 * 60 * 24
        since = time.time() - os.path.getmtime(VERSION_CHECK_FNAME)
        if 0 < since < day:
            if verbose:
                hours = since / 60 / 60
                io.tool_output(f"Too soon to check version: {hours:.1f} hours")
            return

    # To keep startup fast, avoid importing this unless needed
    import requests

    try:
        response = requests.get("https://pypi.org/pypi/stella-chat/json")
        data = response.json()
        latest_version = data["info"]["version"]
        current_version = stella_ai.__version__

        if just_check or verbose:
            io.tool_output(f"Current version: {current_version}")
            io.tool_output(f"Latest version: {latest_version}")

        is_update_available = packaging.version.parse(latest_version) > packaging.version.parse(
            current_version
        )
    except Exception as err:
        io.tool_error(f"Error checking pypi for new version: {err}")
        return False
    finally:
        VERSION_CHECK_FNAME.parent.mkdir(parents=True, exist_ok=True)
        VERSION_CHECK_FNAME.touch()

    ###
    # is_update_available = True

    if just_check or verbose:
        if is_update_available:
            io.tool_output("Update available")
        else:
            io.tool_output("No update available")

    if just_check:
        return is_update_available

    if not is_update_available:
        return False

    install_upgrade(io, latest_version)
    return True

