#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** gguf_info.py is a useful tool that inspects tensors from  **#
#** GGUF files.                                               **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Mar-07-2026 -------------------- **#
#** ********************************************************* **#
#**                                                           **#
#** Author: Thireus <gguf@thireus.com>                        **#
#**                                                           **#
#** https://gguf.thireus.com/                                 **#
#** Thireus' GGUF Tool Suite - Quantize LLMs Like a Chef       **#
#**                                  ·     ·       ·~°          **#
#**     Λ,,Λ             ₚₚₗ  ·° ᵍᵍᵐˡ   · ɪᴋ_ʟʟᴀᴍᴀ.ᴄᴘᴘ°   ᴮᶠ¹⁶ ·  **#
#**    (:·ω·)       。··°      ·   ɢɢᴜғ   ·°·  ₕᵤ𝓰𝓰ᵢₙ𝓰𝒻ₐ𝒸ₑ   ·°   **#
#**    /    o―ヽニニフ))             · · ɪǫ3_xxs      ~·°        **#
#**    し―-J                                                   **#
#**                                                           **#
#** Copyright © 2026 - Thireus.                ₜₑₙₛₒᵣ ₛₑₐₛₒₙᵢₙ𝓰 **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Requirements:
# If using ik_llama.cpp: pip install "gguf @ git+https://github.com/ikawrakow/ik_llama.cpp.git@main#subdirectory=gguf-py" --force; pip install sentencepiece numpy==1.26.4

import sys
# Preserve the original argv immediately so re-execs keep the user's flags.
_ORIG_ARGV = sys.argv[:]  # <-- important: used when re-execing so flags (e.g. --venv) aren't lost
from pathlib import Path
import tempfile
import os

# ---------------------------------------------------------------------
# New bootstrapping/portable-venv code begins here.
# Goal: if required packages are not importable in the current Python,
# create a persistent venv in the system temp folder (so it behaves like
# a temporary-but-persistent space), install the needed packages there,
# and re-execute this script using that venv's Python. The venv dir is
# intentionally not removed so subsequent runs reuse it.
#
# This block intentionally does not remove or alter any existing comments
# in the original script; it only adds code above the original import of
# GGUFReader.
# ---------------------------------------------------------------------

# New CLI flags:
#  -v / --verbose  : show informational/bootstrap logs (when absent, script is quiet except the final "=== Tensors ..." section)
#  --venv          : use the portable venv bootstrap behavior (install/check deps and re-exec into venv if needed)
#  -u / --update   : force update/bootstrapping even if a .verified marker exists
#
# NOTE: we intentionally default to NOT using the venv (so the script behaves
# like it used to historically). The user must opt-in with --venv to enable
# the bootstrap flow. This preserves prior behavior for users who manage
# dependencies themselves.

# Default: quiet mode (no informational/bootstrap logs). Verbose enabled with -v/--verbose.
QUIET = True
USE_VENV = False
FORCE_UPDATE = False

# Parse and remove our bootstrap flags early so they don't interfere with later
# positional argument handling. Preserve -h/--help handling further down.
_new_argv = []
for a in sys.argv:
    if a in ("-v", "--verbose"):
        QUIET = False
    elif a == "--venv":
        USE_VENV = True
    elif a in ("-u", "--update"):
        FORCE_UPDATE = True
    else:
        _new_argv.append(a)
sys.argv[:] = _new_argv

import subprocess
import venv
import shutil
import getpass
import hashlib
import platform
import time
import traceback
import errno

# Packages we need. Keep these in sync with the README/requirements.
_BOOTSTRAP_PACKAGES = [
    'sentencepiece',
    'numpy==1.26.4',
    'gguf @ git+https://github.com/ikawrakow/ik_llama.cpp.git@main#subdirectory=gguf-py'
]

def _info(msg: str):
    """Print informational messages to stderr unless QUIET is set."""
    if not QUIET:
        try:
            sys.stderr.write("[gguf_info.py] "+msg)
            if not msg.endswith("\n"):
                sys.stderr.write("\n")
        except Exception:
            pass

def _error(msg: str):
    """Always print errors to stderr (we keep errors visible even in quiet mode)."""
    try:
        sys.stderr.write("[gguf_info.py] "+msg)
        if not msg.endswith("\n"):
            sys.stderr.write("\n")
    except Exception:
        pass

def _is_importable(module_name: str) -> bool:
    """Return True if module_name can be imported without raising ImportError."""
    try:
        __import__(module_name)
        return True
    except Exception:
        return False

def _get_base_dir() -> Path:
    """
    Return the base directory used for persistent files related to gguf_info.
    Location: <system-temp>/gguf_info_envs or fallback to ./.gguf_info_envs
    """
    base = Path(tempfile.gettempdir()) / "gguf_info_envs"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        base = Path.cwd() / ".gguf_info_envs"
        base.mkdir(parents=True, exist_ok=True)
    return base

def _get_persistent_venv_dir() -> Path:
    """
    Return a path to a persistent-but-temporary venv directory.

    Location: <system-temp>/gguf_info_envs/gguf_info_venv_py{maj}{min}_{owner_hash}
    The owner_hash helps avoid collisions on multi-user systems.
    """
    base = _get_base_dir()

    owner = None
    try:
        uid = getattr(os, "getuid", None)
        if callable(uid):
            owner = str(uid())
        else:
            owner = getpass.getuser()
    except Exception:
        owner = getpass.getuser()

    # Short stable hash combining owner and machine id (platform.node()) to reduce collisions.
    identifier = f"{owner}-{platform.node()}"
    owner_hash = hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:8]

    venv_name = f"gguf_info_venv_py{sys.version_info.major}{sys.version_info.minor}_{owner_hash}"
    return base / venv_name

def _venv_python_path(venv_dir: Path) -> Path:
    """Return the path to the python executable inside the venv for current OS."""
    if os.name == "nt":
        candidate = venv_dir / "Scripts" / "python.exe"
        if candidate.exists():
            return candidate
        return venv_dir / "Scripts" / "python"
    else:
        # Use 'python' if created by venv; sometimes there is 'python3' symlink too.
        candidate = venv_dir / "bin" / "python"
        if candidate.exists():
            return candidate
        return venv_dir / "bin" / "python3"

def _create_venv(venv_dir: Path) -> None:
    """Create a venv at venv_dir with pip available."""
    # Use the stdlib venv to create the environment.
    try:
        venv.create(str(venv_dir), with_pip=True, clear=False, symlinks=True)
    except TypeError:
        # Older Python versions may not accept some args; fallback to simple create
        venv.create(str(venv_dir), with_pip=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create virtualenv at {venv_dir}: {e}")

def _pip_install(venv_python: Path, packages):
    """Run pip install using the venv's python; raises on failure.

    When QUIET is set, pip output is suppressed to avoid spamming stdout/stderr.
    """
    # Ensure pip/setuptools/wheel are up-to-date first
    cmd_upgrade = [str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
    cmd_install = [str(venv_python), "-m", "pip", "install"] + list(packages)

    try:
        if QUIET:
            subprocess.check_call(cmd_upgrade, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.check_call(cmd_install, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.check_call(cmd_upgrade)
            subprocess.check_call(cmd_install)
    except subprocess.CalledProcessError:
        # Re-raise to be handled by caller with a helpful message
        raise

def _get_script_path() -> str:
    """
    Determine the script path to pass to interpreters when re-execing.
    Prefer the original argv[0] if it points to an existing file; otherwise fall back to __file__.
    Return an absolute path string.
    """
    try:
        cand = Path(_ORIG_ARGV[0]) if _ORIG_ARGV else None
        if cand and cand.exists():
            return str(cand.resolve())
    except Exception:
        pass
    try:
        return str(Path(__file__).resolve())
    except Exception:
        # last resort
        return _ORIG_ARGV[0] if _ORIG_ARGV else sys.argv[0]

def _reexec_with_python(python_path: str):
    """Re‑execute the script using the given python interpreter."""
    script = _get_script_path()
    new_argv = [python_path, script] + (_ORIG_ARGV[1:] if len(_ORIG_ARGV) > 1 else [])

    # Sanity check: make sure the target python exists
    if not Path(python_path).exists():
        _error(f"Target Python executable does not exist: {python_path}")
        sys.exit(1)

    try:
        # On Windows (including Cygwin), subprocess works better than os.execv
        if sys.platform == 'win32':
            if not QUIET:
                _info(f"Launching {python_path} with subprocess and exiting.")
            proc = subprocess.run(new_argv)
            sys.exit(proc.returncode)
        else:
            os.execv(python_path, new_argv)
    except Exception as e:
        _error(f"Failed to execute {python_path}: {e}")
        sys.exit(1)

def _ensure_python_version_or_reexec():
    """
    Ensure we're running on Python >= 3.8. If not, attempt to locate a suitable
    interpreter on the system and re-execute this script using it.

    This is required so the persistent venv we create/use is based on a Python
    version >= 3.8 as requested.
    """
    # If already OK, nothing to do.
    if sys.version_info >= (3, 8):
        return

    # If we've already attempted a re-exec, avoid infinite loops.
    if os.environ.get("GGUF_INFO_PY_REEXEC"):
        _error(f"Error: Current Python {sys.version_info.major}.{sys.version_info.minor} is < 3.8 and a re-exec was already attempted.")
        sys.exit(1)

    # Candidate interpreter names to try (ordered prefer newer versions).
    candidates = [
        "python3.16", "python3.15", "python3.14", "python3.13", "python3.12", "python3.11", "python3.10", "python3.9", "python3.8",
        "python3", "python"
    ]

    for cand in candidates:
        path = shutil.which(cand)
        if not path:
            continue
        try:
            out = subprocess.check_output(
                [path, "-c", "import sys; sys.stdout.write(f'{sys.version_info[0]} {sys.version_info[1]}')"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5
            ).strip()
            if not out:
                continue
            major, minor = map(int, out.split())
            if (major, minor) >= (3, 8):
                # Re-exec using this interpreter. Mark env so we don't loop.
                if not QUIET:
                    _info(f"Re-execing with interpreter: {path} (version {major}.{minor})")
                os.environ["GGUF_INFO_PY_REEXEC"] = "1"
                _reexec_with_python(path)
        except Exception:
            # Ignore and try next candidate
            continue

    # If we reach here, no suitable interpreter found.
    _error("Error: Python >= 3.8 is required but no suitable interpreter was found on PATH.")
    _error("Please install Python 3.8+ or run this script with a suitable python (e.g. /usr/bin/python3.10).")
    sys.exit(1)

# ---------------------------------------------------------------------
# Locking helpers for venv creation/update
# ---------------------------------------------------------------------

_LOCK_TIMEOUT = 600  # seconds after which a lock is considered stale
_LOCK_RETRY_INTERVAL = 1  # seconds between retries
_LOCK_MAX_RETRIES = 600  # roughly 10 minutes total (600 * 1)

def _get_lock_path(venv_dir: Path) -> Path:
    """Return path to lock file associated with this venv directory."""
    base = _get_base_dir()
    # Use the venv directory name as part of the lock to avoid cross-venv contention
    return base / f"{venv_dir.name}.lock"

def _write_lock(lock_path: Path) -> None:
    """Write current PID and timestamp to the lock file."""
    content = f"{os.getpid()}\n{time.time()}\n"
    try:
        with open(lock_path, "w") as f:
            f.write(content)
    except Exception as e:
        raise RuntimeError(f"Failed to write lock file {lock_path}: {e}")

def _read_lock(lock_path: Path):
    """Read (pid, timestamp) from lock file. Return (pid, timestamp) or (None, None) on error."""
    try:
        with open(lock_path, "r") as f:
            lines = f.readlines()
        if len(lines) >= 2:
            return int(lines[0].strip()), float(lines[1].strip())
    except Exception:
        pass
    return None, None

def _is_stale(pid, timestamp):
    """Return True if the lock with given pid and timestamp is stale."""
    if pid is None or timestamp is None:
        return True
    if time.time() - timestamp > _LOCK_TIMEOUT:
        return True
    try:
        os.kill(pid, 0)  # signal 0 just checks existence
        return False      # process exists
    except OSError:
        return True       # process does not exist or we lack permission

def _is_lock_stale(lock_path: Path) -> bool:
    """Return True if lock is older than _LOCK_TIMEOUT, or if the owning process no longer exists (Unix only)."""
    pid, timestamp = _read_lock(lock_path)
    return _is_stale(pid, timestamp)

def _acquire_lock(lock_path: Path, wait: bool = True) -> bool:
    """
    Try to acquire the lock. If wait is True, block until lock is acquired.
    Returns True if lock acquired, False if wait=False and lock is held by another.
    Raises exception on unrecoverable error.
    """
    retries = 0
    while True:
        try:
            # Try to create the lock file exclusively
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(f"{os.getpid()}\n{time.time()}\n")
            # Lock acquired
            return True
        except OSError as e:
            if e.errno != errno.EEXIST:
                # Some other error (permissions, etc.) – propagate
                raise
            # Lock exists; read its content to check staleness
            pid, timestamp = _read_lock(lock_path)
            if _is_stale(pid, timestamp):
                _error(f"Removing stale lock file {lock_path} (process {pid} no longer exists or lock too old).")
                try:
                    os.unlink(str(lock_path))
                except OSError:
                    pass
                continue
            if not wait:
                return False
            # Wait and retry
            retries += 1
            if retries > _LOCK_MAX_RETRIES:
                raise TimeoutError(f"Could not acquire lock after {_LOCK_MAX_RETRIES} retries")
            time.sleep(_LOCK_RETRY_INTERVAL)

def _waitfor_lock(lock_path: Path, check_interval: float = 0.1) -> None:
    """
    Wait until the lock file at `lock_path` is released.

    The lock is considered released when:
      - The lock file does not exist, or
      - The lock file exists but the process that created it is dead (stale).
        In that case, the stale file is removed automatically.

    The function does **not** acquire the lock – it only monitors the file.

    Args:
        lock_path: Path to the lock file.
        check_interval: Seconds to sleep between polls.

    Raises:
        OSError: For filesystem errors other than the lock file disappearing.
    """
    while True:
        # If the lock file is gone, we're done.
        if not lock_path.exists():
            return

        # Check if the existing lock file is stale.
        if _is_lock_stale(lock_path):
            # Remove the stale lock file. If it disappears between the check and
            # the removal, ignore the error – that means the lock is already gone.
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            # After removal, consider the lock released. Even if a new lock is
            # created immediately, the next loop iteration will see it and wait.
            return

        # Lock still validly held – wait and retry.
        time.sleep(check_interval)

def _release_lock(lock_path: Path) -> None:
    """Release the lock if we hold it (i.e., if the lock file contains our PID)."""
    try:
        if lock_path.exists():
            pid, _ = _read_lock(lock_path)
            if pid == os.getpid():
                os.unlink(str(lock_path))
    except Exception:
        # Best effort
        pass

# ---------------------------------------------------------------------
# New bootstrapping/portable-venv code continues.
# ---------------------------------------------------------------------

# Helper utilities for reading/writing a multi-entry verified marker.
# The marker format is intentionally simple and backward-compatible:
# - Lines of the form "<creator_python_path>:<resolved_creator_path>:<venv_python_path>"
#   (two‑part lines from older versions are also accepted, resolved = creator)
# - Lines starting with '#' are comments and ignored
def _parse_verified_marker(marker_path: Path):
    """
    Parse the verified marker file.
    Returns a list of tuples (creator_path, resolved_path, venv_python_path).
    For two‑part lines, resolved_path is set equal to creator_path.
    """
    entries = []
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split(":")
                if len(parts) == 2:
                    creator, venv = parts
                    entries.append((creator.strip(), creator.strip(), venv.strip()))
                elif len(parts) >= 3:
                    creator, resolved, venv = parts[0], parts[1], parts[2]
                    entries.append((creator.strip(), resolved.strip(), venv.strip()))
                # else ignore malformed lines
    except Exception:
        # If parsing fails, return empty list (behave conservatively and re-bootstrap)
        return []
    return entries

def _write_or_update_verified_marker(marker_path: Path, creator_path: str, resolved_path: str, venv_path: str):
    """
    Update or create the verified marker, adding/updating the entry:
      creator_path:resolved_path:venv_path
    Preserve other existing entries.
    """
    entries = []
    try:
        entries = _parse_verified_marker(marker_path)
    except Exception:
        entries = []

    # Remove any existing entry for the same creator_path (or same resolved_path? we'll match by creator)
    entries = [e for e in entries if e[0] != creator_path]

    # Add new entry
    entries.append((creator_path, resolved_path, venv_path))

    # Write entries back (sorted for determinism)
    try:
        with open(marker_path, "w", encoding="utf-8") as f:
            for creator, resolved, venv in sorted(entries, key=lambda x: x[0]):
                f.write(f"{creator}:{resolved}:{venv}\n")
            f.write(f"# verified: {time.time()}\n")
    except Exception:
        # best-effort; ignore failures here (non-fatal)
        pass

def _venv_python_version(venv_python: str):
    """Return 'MAJOR.MINOR' string for the given python executable, or None on error."""
    try:
        out = subprocess.check_output([venv_python, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"], text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        return out
    except Exception:
        return None

def _find_matching_venv_from_marker(marker_path: Path):
    """
    Given the marker file, try to find the best venv python to re-exec into for the current interpreter.
    Preference order:
     1) exact mapping keyed by current sys.executable (creator field)
     2) exact mapping keyed by resolved path of current sys.executable (resolved field)
     3) any mapping whose venv_python exists and whose python major.minor matches current interpreter
     4) any mapping whose venv_python exists (last resort)
    Returns path string to venv python to re-exec, or None if none suitable.
    """
    entries = _parse_verified_marker(marker_path)
    if not entries:
        return None

    cur_exec = str(sys.executable)
    cur_real = os.path.realpath(cur_exec)
    cur_ver = f"{sys.version_info[0]}.{sys.version_info[1]}"

    # 1) exact match on creator
    for creator, resolved, venv in entries:
        if creator == cur_exec:
            if Path(venv).exists():
                return venv

    # 2) exact match on resolved
    for creator, resolved, venv in entries:
        if resolved == cur_real:
            if Path(venv).exists():
                return venv

    # 3) match by major.minor version of python executable inside venv
    for creator, resolved, venv in entries:
        if Path(venv).exists():
            vver = _venv_python_version(venv)
            if vver == cur_ver:
                return venv

    # 4) any existing venv python (last resort)
    for creator, resolved, venv in entries:
        if Path(venv).exists():
            return venv

    return None

def ensure_dependencies_available_or_reexec():
    """
    Ensure required dependencies are importable. If not, create a
    persistent venv in the system temp folder, install packages there,
    and re-exec this script with the venv Python. This function returns
    only when dependencies are available in the current interpreter.
    """
    # Quick check: can we import the gguf package we need?
    # the gguf package exposes gguf.gguf_reader, which is the primary import we need.
    if _is_importable("gguf.gguf_reader"):
        return  # everything already available; proceed

    venv_dir = _get_persistent_venv_dir()
    vpython = _venv_python_path(venv_dir)
    installed_flag = venv_dir / ".installed"
    verified_marker = _get_base_dir() / ".verified"
    lock_path = _get_lock_path(venv_dir)

    # NEW: If another process is currently creating/updating the venv (lock exists and not stale),
    # wait on that lock BEFORE attempting to inspect/install. This prevents concurrent processes
    # from racing to install packages into the same venv.
    try:
        if lock_path.exists() and (not _is_lock_stale(lock_path)):
            owner_pid, _ = _read_lock(lock_path)
            if owner_pid and owner_pid != os.getpid():
                _info("Detected ongoing bootstrap by another process; waiting for it to finish...")
                # Wait for the other creator to finish by acquiring the lock (this will block until it is released)
                _waitfor_lock(lock_path)
                # At this point the other process has released the lock.
                # Re-check environment state now that the creator finished.
                try:
                    # If import is now available, we are done.
                    if _is_importable("gguf.gguf_reader"):
                        return
                    # If the installed marker exists and venv python exists, re-exec into it rather than trying to install.
                    if installed_flag.exists() and vpython.exists():
                        _info("Bootstrap appears complete after waiting; re-execing into venv instead of installing.")
                        # mark re-exec so we don't loop
                        os.environ["GGUF_INFO_PY_REEXEC"] = "1"
                        os.environ["GGUF_INFO_CREATOR_PYTHON"] = os.environ.get("GGUF_INFO_CREATOR_PYTHON", str(sys.executable))
                        # release lock before exec (the creating process already finished; we can release ours)
                        _reexec_with_python(str(vpython))
                        # if exec returns, continue to normal flow (unlikely)
                    # Otherwise, fall through to the flow that will create/install the venv.
                except Exception:
                    pass
                    # and continue to normal flow (we will attempt create/install)
                # If we reach here without returning or re-execing, proceed to create/install path below.
    except Exception:
        # If lock checks fail, continue and attempt normal flow (best-effort)
        pass

    # Helper to re-exec if we are not already inside the target venv
    def maybe_reexec_into_venv(target_python: str):
        # Avoid re-exec loops: only re-exec if target_python exists, differs from current exec,
        # and we have not already re-exec'ed in this process lifetime.
        try:
            if target_python and Path(target_python).exists():
                same = False
                try:
                    same = Path(sys.executable).samefile(Path(target_python))
                except Exception:
                    # if samefile fails, fall back to string compare
                    same = str(sys.executable) == str(target_python)
                if same:
                    _info("Current interpreter is already the target venv python; not re-execing.")
                    return
                if os.environ.get("GGUF_INFO_PY_REEXEC"):
                    _info("Already re-execed once; not re-execing again.")
                    return
                _info(f"Re-executing into existing venv at {target_python}")
                # mark we've re-execed so we don't attempt again
                os.environ["GGUF_INFO_PY_REEXEC"] = "1"
                # preserve creator info
                os.environ["GGUF_INFO_CREATOR_PYTHON"] = os.environ.get("GGUF_INFO_CREATOR_PYTHON", str(sys.executable))
                _reexec_with_python(target_python)
        except Exception as e:
            # If re-exec fails, log and continue (we will attempt other flows)
            _info(f"Re-exec attempt failed: {e}")

    # Case 1: we are already inside the target venv (current interpreter is the venv's python)
    if vpython.exists() and Path(sys.executable).samefile(vpython):
        # We are in the venv, but import failed. Need to (re)install packages.
        # Wait for lock (wait indefinitely)
        _info("Waiting for lock release...")
        _waitfor_lock(lock_path)

        # After waiting, **re-check** state: maybe someone else already installed while we were waiting to acquire lock.
        if _is_importable("gguf.gguf_reader"):
            _info("gguf already importable; skipping install.")
            return

        if installed_flag.exists():
            # installed marker present — assume packages installed; proceed
            _info("Installed marker present; skipping install and continuing.")
            # mark re-exec so we don't loop
            os.environ["GGUF_INFO_PY_REEXEC"] = "1"
            os.environ["GGUF_INFO_CREATOR_PYTHON"] = os.environ.get("GGUF_INFO_CREATOR_PYTHON", str(sys.executable))
            _reexec_with_python(str(vpython))
            # if exec returns, continue to normal flow (unlikely)
            return

        # Acquire lock (wait indefinitely)
        _info("Acquiring lock to install packages into current venv...")
        _acquire_lock(lock_path, wait=True)
        try:
            # Install packages
            _info("Installing required packages into existing venv (this may take a while)...")
            _pip_install(vpython, _BOOTSTRAP_PACKAGES)
            # Mark as installed
            installed_flag.touch()
            # Set environment variable to indicate that we performed an install in this bootstrap session
            os.environ["GGUF_INFO_DID_INSTALL"] = "1"
        except Exception as e:
            _error(f"Failed to install packages in venv: {e}")
            _release_lock(lock_path)
            sys.exit(1)

        # Important: DO NOT release lock here. We must hold the lock across the re-exec
        # so other concurrent processes will wait until the verified marker is written.
        # Store lock path in environment so the re-exec'ed process will be able to release it
        os.environ["GGUF_INFO_LOCK_PATH"] = str(lock_path)
        # Record the creator python path so the re-exec'ed process can add an entry to the verified marker.
        os.environ["GGUF_INFO_CREATOR_PYTHON"] = os.environ.get("GGUF_INFO_CREATOR_PYTHON", str(sys.executable))
        _info("Packages installed; re-executing to refresh environment (holding lock until verified marker is written)...")
        os.environ["GGUF_INFO_PIP_REEXEC"] = "1"
        # mark re-exec so we don't loop
        os.environ["GGUF_INFO_PY_REEXEC"] = "1"
        _reexec_with_python(str(vpython))

    # Case 2: not inside the target venv
    else:
        # If venv already exists and is marked installed, attempt to re-use it if marker points to a candidate.
        if vpython.exists() and installed_flag.exists():
            try:
                if verified_marker.exists():
                    candidate = _find_matching_venv_from_marker(verified_marker)
                    if candidate:
                        maybe_reexec_into_venv(candidate)
                    else:
                        _info("Verified marker does not contain a venv suitable for this interpreter; will (re)create/use venv for this python.")
                else:
                    _info("Venv installed but verified marker not present; waiting for any creator to finish.")
            except Exception:
                _info("Error reading verified marker; will acquire lock and wait.")

            # If there's a lock from another process, wait for it as well
            try:
                if lock_path.exists() and (not _is_lock_stale(lock_path)):
                    owner_pid, _ = _read_lock(lock_path)
                    if owner_pid and owner_pid != os.getpid():
                        _info("Detected ongoing bootstrap by another process; waiting for it to finish...")
                        _waitfor_lock(lock_path)
                        # After acquiring lock, re-check install state exactly like earlier:
                        try:
                            if _is_importable("gguf.gguf_reader"):
                                _info("gguf importable after waiting; returning.")
                                return
                            if installed_flag.exists() and vpython.exists():
                                _info("Installed marker present after waiting; re-execing into venv.")
                                maybe_reexec_into_venv(str(vpython))
                            # else: fall through to create/install path
                        except Exception:
                            pass
            except Exception:
                pass

        # Otherwise, we need to create/update the venv. Acquire lock.
        lock_path = _get_lock_path(venv_dir)
        _info("Acquiring lock to create/update venv...")
        _acquire_lock(lock_path, wait=True)

        # After acquiring, check again if the venv is now installed (another process might have finished while we waited)
        if vpython.exists() and installed_flag.exists():
            _info("Venv was installed by another process while waiting for lock; releasing lock and re-executing if appropriate.")
            _release_lock(lock_path)
            maybe_reexec_into_venv(str(vpython))

        try:
            # Create venv if it doesn't exist
            if not venv_dir.exists():
                _info(f"Creating persistent virtual environment at: {venv_dir}")
                _create_venv(venv_dir)

            # Ensure we have a python executable
            vpython = _venv_python_path(venv_dir)
            if not vpython.exists():
                raise RuntimeError(f"Python executable not found in venv {venv_dir}")

            # Install packages
            _info(f"Installing required packages into venv at {venv_dir} (this may take a while)...")
            _pip_install(vpython, _BOOTSTRAP_PACKAGES)
            # Mark as installed
            installed_flag.touch()
            # Set environment variable to indicate that we performed an install in this bootstrap session
            os.environ["GGUF_INFO_DID_INSTALL"] = "1"
        except Exception as e:
            _error(f"Failed to create/install venv: {e}")
            _release_lock(lock_path)
            sys.exit(1)

        # Important: DO NOT release the lock here. Hold it across re-exec so other concurrent processes block
        # until the verified marker is created by the re-executed process. Save the lock path in the environment.
        os.environ["GGUF_INFO_LOCK_PATH"] = str(lock_path)
        # Save who the creator was (the interpreter currently running this code)
        os.environ["GGUF_INFO_CREATOR_PYTHON"] = os.environ.get("GGUF_INFO_CREATOR_PYTHON", str(sys.executable))
        _info("Venv ready; re-executing into it (holding lock until verified marker is written)...")
        os.environ["GGUF_INFO_PIP_REEXEC"] = "1"
        # mark re-exec so we don't loop
        os.environ["GGUF_INFO_PY_REEXEC"] = "1"
        _reexec_with_python(str(vpython))

# Ensure Python version >= 3.8 and dependency bootstrapping only if user requested --venv.
if USE_VENV:
    marker = _get_base_dir() / ".verified"

    if marker.exists() and not FORCE_UPDATE:
        # New behavior: .verified contains lines with three fields.
        # We will prefer a candidate venv for this interpreter. Re-exec only if:
        #  - candidate exists, and
        #  - candidate is not the same as current sys.executable, and
        #  - we have not already re-exec'ed in this process.
        try:
            candidate = _find_matching_venv_from_marker(marker)
            if candidate and Path(candidate).exists():
                # # Check if candidate is same as current interpreter
                # same = False
                # try:
                #     same = Path(sys.executable).samefile(Path(candidate))
                # except Exception:
                #     same = str(sys.executable) == str(candidate)

                # if same:
                #     _info("Found verified venv candidate that matches current interpreter; skipping re-exec.")
                # else:
                if not os.environ.get("GGUF_INFO_PY_REEXEC"):
                    _info(f"Found verified venv candidate for this interpreter; re-execing into {candidate} to ensure environment is active.")
                    # Preserve the creator info
                    os.environ["GGUF_INFO_CREATOR_PYTHON"] = os.environ.get("GGUF_INFO_CREATOR_PYTHON", str(sys.executable))
                    os.environ["GGUF_INFO_PY_REEXEC"] = "1"
                    _reexec_with_python(candidate)
                else:
                    _info("Already re-execed once; proceeding.")
            else:
                # No suitable entry for this interpreter; fall back to full bootstrap flow.
                _info("No verified venv entry suitable for this interpreter; performing bootstrap flow (if needed).")
                _ensure_python_version_or_reexec()
                ensure_dependencies_available_or_reexec()
        except Exception as e:
            _info(f"Error reading marker: {e}; will recreate.")
            try:
                marker.unlink()
            except Exception:
                pass
            _ensure_python_version_or_reexec()
            ensure_dependencies_available_or_reexec()
    else:
        # No marker, or --update was given: perform full bootstrap
        _ensure_python_version_or_reexec()
        ensure_dependencies_available_or_reexec()

# ---------------------------------------------------------------------
# New bootstrapping/portable-venv code ends here.
# ---------------------------------------------------------------------

# import the GGUF reader
# NOTE: if the user didn't request --venv and the import fails, we do NOT auto-install;
# instead we print a helpful error instructing the user to either install dependencies
# into their environment or re-run with --venv to let the script manage them.
try:
    from gguf.gguf_reader import GGUFReader
except Exception as e:
    if not USE_VENV:
        _error("Error: failed to import 'gguf.gguf_reader' from the current Python environment.")
        _error("If you want this script to manage a portable venv and install dependencies automatically, re-run with the --venv option.")
        _error("Otherwise, install the required packages into your environment, e.g.:")
        _error("  pip install sentencepiece numpy==1.26.4")
        _error("  pip install \"gguf @ git+https://github.com/ikawrakow/ik_llama.cpp.git@main#subdirectory=gguf-py\"")
        # Show the original exception only in verbose mode
        if not QUIET:
            traceback.print_exc()
        sys.exit(1)
    else:
        # If USE_VENV was requested but import still failed here, re-raise so the bootstrap
        # flow (which may have re-exec'd) can handle it, or show a verbose traceback.
        _error("Error: import failed even though --venv was requested. Full traceback follows (verbose mode only).")
        if not QUIET:
            traceback.print_exc()
        sys.exit(1)

def print_help(prog_name: str):
    """Print help / usage information."""
    help_text = f"""Usage: {prog_name} [OPTIONS] [path/to/model.gguf]
Inspect tensors in a GGUF file.

Options:
  -h, --help      Show this help message and exit.
  -               Read GGUF bytes from stdin (explicit).
  -v, --verbose   Show informational/bootstrap logs (default: quiet).
  --venv          Use the portable venv bootstrap behaviour (install/check deps).
  -u, --update    Force update/bootstrapping even if a .verified marker exists.
If no path is given and stdin is not a TTY, the program will read piped bytes from stdin.
Examples:
  {prog_name} model.gguf
  cat model.gguf | {prog_name}
  {prog_name} -
  {prog_name} --venv model.gguf     # let the script ensure dependencies in a portable venv
"""
    print(help_text, file=sys.stdout)

def main():
    # This script now accepts either:
    #  - ./gguf_info.py path/to/model.gguf         (unchanged)
    #  - ./gguf_info.py -                         (explicitly read stdin)
    #  - cat model.gguf | ./gguf_info.py          (read piped bytes from stdin; no args)
    #
    # If neither a path nor piped data is provided, it prints the usage message (same as before).
    temp_file_path = None

    # If user asked for help explicitly, show it and exit
    if len(sys.argv) >= 2:
        if sys.argv[1] in ("-h", "--help"):
            print_help(sys.argv[0])
            sys.exit(0)
        # also respond to help request anywhere in args
        for a in sys.argv[1:]:
            if a in ("-h", "--help"):
                print_help(sys.argv[0])
                sys.exit(0)

    # Determine input source
    if len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg == "-":
            # Explicit request to read from stdin
            try:
                data = sys.stdin.buffer.read()
            except Exception:
                data = None

            if not data:
                print_help(sys.argv[0])
                sys.exit(1)

            # Write piped bytes to a temporary file so GGUFReader can open it by path
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".gguf")
            try:
                tmp.write(data)
                tmp.flush()
                tmp.close()
                temp_file_path = Path(tmp.name)
                gguf_path = temp_file_path
            except Exception:
                try:
                    tmp.close()
                except Exception:
                    pass
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
                print("[gguf_info.py] "+f"Error: failed to read stdin into temporary file.", file=sys.stderr)
                sys.exit(1)
        else:
            gguf_path = Path(arg)

    elif len(sys.argv) == 1:
        # No filename arg provided. If there's piped data on stdin, read it.
        if not sys.stdin.isatty():
            try:
                data = sys.stdin.buffer.read()
            except Exception:
                data = None

            if not data:
                print_help(sys.argv[0])
                sys.exit(1)

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".gguf")
            try:
                tmp.write(data)
                tmp.flush()
                tmp.close()
                temp_file_path = Path(tmp.name)
                gguf_path = temp_file_path
            except Exception:
                try:
                    tmp.close()
                except Exception:
                    pass
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
                print("[gguf_info.py] "+f"Error: failed to read stdin into temporary file.", file=sys.stderr)
                sys.exit(1)
        else:
            print_help(sys.argv[0])
            sys.exit(1)
    else:
        # Too many arguments
        # If user provided -h/--help among many args earlier, we would have exited.
        print_help(sys.argv[0])
        sys.exit(1)

    # At this point gguf_path should be set to a Path (either user-supplied or temporary file)
    try:
        reader = GGUFReader(gguf_path)   # loads metadata & tensor index :contentReference[oaicite:0]{index=0}

        # If we reached this point, the script successfully imported and used GGUFReader.
        # We update the verified marker only if we actually performed an install/update
        # in this run (i.e., if FORCE_UPDATE was given or if the environment variable
        # GGUF_INFO_DID_INSTALL is set). This prevents unnecessary marker updates on
        # every successful run.
        if USE_VENV and (FORCE_UPDATE or os.environ.get("GGUF_INFO_DID_INSTALL")):
            try:
                base = _get_base_dir()
                marker = base / ".verified"
                with open(marker, "a", encoding="utf-8"):
                    pass  # ensure file exists
                # Determine creator path and resolved path
                creator = os.environ.get("GGUF_INFO_CREATOR_PYTHON", None)
                venv_python = str(sys.executable)
                if creator:
                    # The creator is the interpreter that performed the bootstrap.
                    # We store both its original path and its resolved path.
                    creator_path = creator
                    resolved_creator = os.path.realpath(creator_path)
                else:
                    # No creator info; use current interpreter as creator (should not happen in bootstrap flows)
                    creator_path = venv_python
                    resolved_creator = os.path.realpath(creator_path)

                _write_or_update_verified_marker(marker, creator_path, resolved_creator, venv_python)
                _info(f"Wrote/updated verified marker at {marker} (install/update performed).")
            except Exception:
                if not QUIET:
                    _info("Warning: could not write verified marker (non-fatal).")
            # If we were holding a lock across re-exec, release it now since marker is written.
            lock_path_env = os.environ.get("GGUF_INFO_LOCK_PATH")
            if lock_path_env:
                try:
                    _release_lock(Path(lock_path_env))
                    _info("Released bootstrap lock after writing verified marker.")
                except Exception:
                    _info("Warning: failed to release bootstrap lock (non-fatal).")
                # Clear the environment variable to avoid accidental reuse
                try:
                    del os.environ["GGUF_INFO_LOCK_PATH"]
                except Exception:
                    pass

        # Print only the script results. QUIET suppresses non-essential messages,
        # but the "=== Tensors in ... ===" line is the required script output.
        print(f"=== Tensors in {gguf_path.name} ===")
        # reader.tensors is a list of TensorEntry objects :contentReference[oaicite:1]{index=1}
        for tensor in reader.tensors:
            name = tensor.name

            # --- Shape: convert tensor.shape (array-like) into a Python tuple of ints
            try:
                shape = tuple(int(dim) for dim in tensor.shape)
            except Exception:
                shape = tuple(tensor.shape)

            # --- Dtype / quantization type: use the enum name
            # e.g. tensor.tensor_type.name might be "Q8_0", "F16", etc.
            dtype = tensor.tensor_type.name.lower()  # keep uppercase like "Q8_0"; use .lower() if you prefer "q8_0"

            # --- Number of elements:
            if hasattr(tensor, 'n_elements'):
                try:
                    elements = int(tensor.n_elements)
                except Exception:
                    # fallback to computing from shape
                    elements = 1
                    for dim in shape:
                        elements *= dim
            else:
                # compute product of dims
                elements = 1
                for dim in shape:
                    elements *= dim

            # --- Number of bytes:
            if hasattr(tensor, 'n_bytes'):
                try:
                    byte_count = int(tensor.n_bytes)
                except Exception:
                    # fallback to data buffer size
                    try:
                        byte_count = tensor.data.nbytes
                    except Exception:
                        byte_count = None
            else:
                # fallback: if tensor.data is a NumPy array or memmap:
                try:
                    byte_count = tensor.data.nbytes
                except Exception:
                    byte_count = None

            # Format byte_count if None
            byte_str = str(byte_count) if byte_count is not None else "unknown"

            print(f"{name}\tshape={shape}\tdtype={dtype}\telements={elements}\tbytes={byte_str}")

    finally:
        # Clean up temporary file if we created one
        if temp_file_path is not None:
            try:
                os.unlink(str(temp_file_path))
            except Exception:
                # best-effort cleanup; don't raise further errors on cleanup failure
                pass

if __name__ == "__main__":
    main()
