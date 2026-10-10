"""MongoDB client singleton for the Atlas functional dataset.

Connection settings come from the environment (``.env`` is loaded at import):
``MONGODB_URI`` and ``MONGODB_DATABASE``, plus ``SSH_HOST``, ``SSH_PORT``, ``SSH_USERNAME``
and ``SSH_PASSWORD`` when the database is only reachable through an SSH tunnel. The client
and the tunnel are opened lazily on the first ``get_client`` call and shared by the whole
process.
"""

from __future__ import annotations

import atexit
import os
import select
import socket
import threading
from typing import TYPE_CHECKING

import paramiko
from dotenv import load_dotenv
from pymongo import MongoClient

if TYPE_CHECKING:
    from pymongo.database import Database

load_dotenv()

# Process-wide singletons, created by the first get_client() call.
_client: MongoClient | None = None
_tunnel: _SSHTunnel | None = None


class _SSHTunnel:
    """Minimal SSH port-forward tunnel.

    Listens on an ephemeral port on 127.0.0.1 and forwards every accepted connection
    through the SSH server to ``remote_host:remote_port`` (as seen from that server).
    The connection is opened in the constructor, and a daemon thread accepts clients
    until ``stop``.

    Args:
        ssh_host:     SSH server to connect to.
        ssh_port:     its SSH port.
        ssh_username: SSH login, or None for paramiko's default.
        ssh_password: SSH password, or None to rely on keys or an agent.
        remote_host:  forwarding target, resolved on the SSH server.
        remote_port:  forwarding target port.

    Attributes:
        local_bind_port: the local port clients connect to.
    """

    def __init__(
        self,
        ssh_host: str,
        ssh_port: int,
        ssh_username: str | None,
        ssh_password: str | None,
        remote_host: str,
        remote_port: int,
    ) -> None:
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._ssh.connect(
            hostname=ssh_host,
            port=ssh_port,
            username=ssh_username,
            password=ssh_password,
        )
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self.local_bind_port: int = self._server.getsockname()[1]
        self._server.listen(5)
        self._remote_host = remote_host
        self._remote_port = remote_port
        self._active = True
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        """Accept local clients and relay each through a new SSH channel, until stopped.

        The one-second accept timeout lets the loop notice ``stop``. A client whose channel
        cannot be opened is closed and the loop carries on; a lost SSH transport ends it.
        """
        while self._active:
            self._server.settimeout(1.0)
            try:
                client_sock, addr = self._server.accept()
            except (TimeoutError, OSError):
                continue
            transport = self._ssh.get_transport()
            if transport is None:
                break
            try:
                chan = transport.open_channel(
                    "direct-tcpip", (self._remote_host, self._remote_port), addr
                )
            except Exception:
                client_sock.close()
                continue
            threading.Thread(
                target=_relay, args=(client_sock, chan), daemon=True
            ).start()

    def stop(self) -> None:
        """Stop accepting clients and close the listening socket and the SSH connection."""
        self._active = False
        self._server.close()
        self._ssh.close()


def _relay(sock: socket.socket, chan: paramiko.Channel) -> None:
    """Copy bytes both ways between a local socket and an SSH channel until either closes.

    Args:
        sock: the accepted local client socket.
        chan: the ``direct-tcpip`` channel to the remote port.
    """
    while True:
        r, _, _ = select.select([sock, chan], [], [], 1.0)
        if sock in r:
            data = sock.recv(4096)
            if not data:
                break
            chan.send(data)
        if chan in r:
            data = chan.recv(4096)
            if not data:
                break
            sock.send(data)
    sock.close()
    chan.close()


def _start_tunnel() -> None:
    """Open the SSH tunnel from the ``SSH_*`` environment variables.

    The tunnel forwards to MongoDB's default port (27017) on the SSH server itself, and is
    closed at interpreter exit.
    """
    global _tunnel
    _tunnel = _SSHTunnel(
        ssh_host=os.environ["SSH_HOST"],
        ssh_port=int(os.environ.get("SSH_PORT", "22")),
        ssh_username=os.environ.get("SSH_USERNAME"),
        ssh_password=os.environ.get("SSH_PASSWORD"),
        remote_host="127.0.0.1",
        remote_port=27017,
    )
    atexit.register(_tunnel.stop)


def get_client() -> MongoClient:
    """Return the process-wide MongoClient, opening it on first call.

    When ``SSH_HOST`` is set, the first call also opens the SSH tunnel and points the
    client at it by rewriting ``localhost:27017`` in ``MONGODB_URI`` to the tunnel's local
    port.

    Returns:
        The shared ``pymongo.MongoClient``.

    Raises:
        ValueError: if ``MONGODB_URI`` is not set in the environment or ``.env``.
    """
    global _client
    if _client is None:
        if os.environ.get("SSH_HOST") and _tunnel is None:
            _start_tunnel()

        uri = os.environ.get("MONGODB_URI")
        if not uri:
            raise ValueError(
                "MONGODB_URI is not set — add it to .env or the environment"
            )

        if _tunnel is not None:
            uri = uri.replace("localhost:27017", f"127.0.0.1:{_tunnel.local_bind_port}")

        _client = MongoClient(uri)
    return _client


def get_database(name: str | None = None) -> Database:
    """Return a Database handle from the shared client (the pipeline's entry point).

    Args:
        name: database to open, defaults to the ``MONGODB_DATABASE`` environment
            variable.

    Returns:
        The ``pymongo.database.Database`` every discovery and loader call takes
        as its first argument.

    Raises:
        ValueError: if no name is given and ``MONGODB_DATABASE`` is not set.
    """
    db_name = name or os.environ.get("MONGODB_DATABASE")
    if not db_name:
        raise ValueError("MONGODB_DATABASE is not set — pass a name or add it to .env")
    return get_client()[db_name]


def _reset_client() -> None:
    """Reset cached client and tunnel. For tests only."""
    global _client, _tunnel
    if _tunnel is not None:
        _tunnel.stop()
        _tunnel = None
    _client = None
