from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WebsiteProcess:
    process: subprocess.Popen
    web_dir: Path
    port: int


_install_locks: dict[Path, threading.Lock] = {}
_install_locks_guard = threading.Lock()
_active_sites: dict[int, WebsiteProcess] = {}
_active_sites_guard = threading.RLock()
_shutdown_hooks_installed = False


def get_install_lock(web_dir: Path) -> threading.Lock:
    resolved = web_dir.resolve()
    with _install_locks_guard:
        lock = _install_locks.get(resolved)
        if lock is None:
            lock = threading.Lock()
            _install_locks[resolved] = lock
        return lock


def register_site(site: WebsiteProcess) -> None:
    with _active_sites_guard:
        _active_sites[site.process.pid] = site


def unregister_site(site: WebsiteProcess) -> None:
    with _active_sites_guard:
        _active_sites.pop(site.process.pid, None)


def stop_all_websites() -> None:
    with _active_sites_guard:
        sites = list(_active_sites.values())
    for site in sites:
        stop_website(site)


def install_shutdown_hooks() -> None:
    global _shutdown_hooks_installed
    if _shutdown_hooks_installed:
        return
    _shutdown_hooks_installed = True
    atexit.register(stop_all_websites)

    if threading.current_thread() is not threading.main_thread():
        return

    previous_handlers = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }

    def handle_shutdown(signum: int, frame: object) -> None:
        stop_all_websites()
        previous = previous_handlers.get(signum)
        if callable(previous):
            previous(signum, frame)
        elif previous == signal.SIG_DFL:
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)


def is_port_open(port: int, host: str = "localhost", timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def ensure_port_available(port: int, web_dir: Path) -> None:
    if is_port_open(port):
        raise RuntimeError(
            f"Port {port} is already in use before starting website: {web_dir}. "
            "Choose another --website-port or stop the existing process."
        )


def wait_for_port(port: int, timeout: int = 20, process: subprocess.Popen | None = None) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if is_port_open(port):
            return True
        if process is not None and process.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def start_website(web_dir: Path | str, port: int = 5173, install: bool = True) -> WebsiteProcess:
    web_dir = Path(web_dir)
    ensure_port_available(port, web_dir)
    if install:
        with get_install_lock(web_dir):
            subprocess.run(["pnpm", "install"], cwd=web_dir, check=False)
    ensure_port_available(port, web_dir)

    proc = subprocess.Popen(
        ["pnpm", "dev", "--port", str(port), "--strictPort"],
        cwd=web_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    site = WebsiteProcess(proc, web_dir, port)
    register_site(site)
    if not wait_for_port(port, process=proc):
        stop_website(site)
        raise RuntimeError(f"Website failed to start on port {port}: {web_dir}")
    if proc.poll() is not None:
        unregister_site(site)
        raise RuntimeError(f"Website process exited while starting on port {port}: {web_dir}")
    return site


def stop_website(site: WebsiteProcess) -> None:
    proc = site.process
    unregister_site(site)
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5)
        except Exception:
            pass


install_shutdown_hooks()
