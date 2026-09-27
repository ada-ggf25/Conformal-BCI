"""Tests for src/db/client.py.

Nothing here talks to MongoDB or an SSH server: ``MongoClient`` and ``paramiko.SSHClient`` are
replaced, while the tunnel's local listener and its relay run over real loopback sockets, so the
bytes that cross them are the real test.

Accepted gaps: the accept loop's idle one-second timeout and its exit when paramiko reports no
transport (the thread just waits or ends, nothing observable changes), and the ``atexit``
registration itself.
"""

from __future__ import annotations

import socket
import threading
from unittest.mock import MagicMock

import pytest

from src.db import client


@pytest.fixture(autouse=True)
def _fresh_client(monkeypatch):
    """Every test starts with no cached client or tunnel and a clean MongoDB environment."""
    monkeypatch.setattr(client, "_client", None)
    monkeypatch.setattr(client, "_tunnel", None)
    for var in (
        "MONGODB_URI",
        "MONGODB_DATABASE",
        "SSH_HOST",
        "SSH_PORT",
        "SSH_USERNAME",
        "SSH_PASSWORD",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(client.atexit, "register", lambda *_a, **_k: None)


@pytest.fixture
def mongo(monkeypatch):
    """Replace MongoClient with a mock that records the URI it was opened with."""
    mc = MagicMock(name="MongoClient")
    monkeypatch.setattr(client, "MongoClient", mc)
    return mc


class TestGetClient:
    def test_missing_uri_is_an_error(self, mongo):
        """Without MONGODB_URI there is nothing to connect to, so it raises rather than guess."""
        with pytest.raises(ValueError, match="MONGODB_URI"):
            client.get_client()
        mongo.assert_not_called()

    def test_client_is_opened_once_and_reused(self, mongo, monkeypatch):
        """Every caller shares one connection pool, opened on the first call."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://db.example:27017")
        first, second = client.get_client(), client.get_client()
        assert first is second
        mongo.assert_called_once_with("mongodb://db.example:27017")

    def test_ssh_tunnel_rewrites_the_uri_to_the_local_forward(self, mongo, monkeypatch):
        """With SSH_HOST set, localhost:27017 must become the tunnel's local port, or the
        client would connect to a MongoDB on this machine instead of the remote one."""
        tunnel = MagicMock(local_bind_port=50123)
        ctor = MagicMock(return_value=tunnel)
        monkeypatch.setattr(client, "_SSHTunnel", ctor)
        monkeypatch.setenv("SSH_HOST", "gateway.example")
        monkeypatch.setenv("SSH_USERNAME", "lab")
        monkeypatch.setenv("MONGODB_URI", "mongodb://user:pw@localhost:27017/?x=1")
        client.get_client()
        mongo.assert_called_once_with("mongodb://user:pw@127.0.0.1:50123/?x=1")
        kwargs = ctor.call_args.kwargs
        assert kwargs["ssh_host"] == "gateway.example"
        assert kwargs["ssh_port"] == 22  # the SSH default, as an int
        assert kwargs["ssh_username"] == "lab"
        assert (kwargs["remote_host"], kwargs["remote_port"]) == ("127.0.0.1", 27017)

    def test_ssh_port_is_read_from_the_environment(self, mongo, monkeypatch):
        """A non-default SSH port reaches the tunnel as an integer."""
        ctor = MagicMock(return_value=MagicMock(local_bind_port=1))
        monkeypatch.setattr(client, "_SSHTunnel", ctor)
        monkeypatch.setenv("SSH_HOST", "gateway.example")
        monkeypatch.setenv("SSH_PORT", "2222")
        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
        client.get_client()
        assert ctor.call_args.kwargs["ssh_port"] == 2222

    def test_no_tunnel_without_ssh_host(self, mongo, monkeypatch):
        """A direct connection leaves the URI untouched and opens no tunnel."""
        ctor = MagicMock()
        monkeypatch.setattr(client, "_SSHTunnel", ctor)
        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
        client.get_client()
        ctor.assert_not_called()
        mongo.assert_called_once_with("mongodb://localhost:27017")


class TestGetDatabase:
    def test_explicit_name_wins_over_the_environment(self, mongo, monkeypatch):
        """Passing a name opens that database even when MONGODB_DATABASE names another."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://h")
        monkeypatch.setenv("MONGODB_DATABASE", "env_db")
        client.get_database("explicit_db")
        mongo.return_value.__getitem__.assert_called_once_with("explicit_db")

    def test_name_defaults_to_the_environment(self, mongo, monkeypatch):
        """With no argument the database named in MONGODB_DATABASE is opened."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://h")
        monkeypatch.setenv("MONGODB_DATABASE", "env_db")
        client.get_database()
        mongo.return_value.__getitem__.assert_called_once_with("env_db")

    def test_no_name_anywhere_is_an_error(self, mongo, monkeypatch):
        """Reading an unnamed database would silently pick the server default, so it raises."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://h")
        with pytest.raises(ValueError, match="MONGODB_DATABASE"):
            client.get_database()


def test_reset_client_stops_the_tunnel_and_drops_the_cache(monkeypatch):
    """After a reset the next get_client opens a new connection."""
    tunnel = MagicMock()
    monkeypatch.setattr(client, "_tunnel", tunnel)
    monkeypatch.setattr(client, "_client", object())
    client._reset_client()
    tunnel.stop.assert_called_once()
    assert client._client is None and client._tunnel is None


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from a blocking socket."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return buf


def test_reset_without_a_tunnel_only_drops_the_client(monkeypatch):
    """A direct connection has no tunnel to stop; the reset still clears the cache."""
    monkeypatch.setattr(client, "_client", object())
    client._reset_client()
    assert client._client is None and client._tunnel is None


class TestRelay:
    def test_bytes_flow_both_ways_and_both_ends_close(self):
        """The relay copies each side to the other and closes both once one side hangs up."""
        local, local_peer = socket.socketpair()
        chan, chan_peer = socket.socketpair()
        for s in (local_peer, chan_peer):
            s.settimeout(5)
        t = threading.Thread(target=client._relay, args=(local, chan), daemon=True)
        t.start()
        local_peer.sendall(b"query")
        assert _recv_exact(chan_peer, 5) == b"query"
        chan_peer.sendall(b"reply!")
        assert _recv_exact(local_peer, 6) == b"reply!"
        local_peer.close()
        t.join(timeout=5)
        assert not t.is_alive()
        assert chan_peer.recv(1) == b""  # the relay closed its channel end

    def test_remote_hangup_also_ends_the_relay(self):
        """A channel closed by the remote side ends the relay just like a local hangup."""
        local, local_peer = socket.socketpair()
        chan, chan_peer = socket.socketpair()
        local_peer.settimeout(5)
        t = threading.Thread(target=client._relay, args=(local, chan), daemon=True)
        t.start()
        chan_peer.close()
        t.join(timeout=5)
        assert not t.is_alive()
        assert local_peer.recv(1) == b""


class _FakeTransport:
    """Stands in for a paramiko transport: each channel is one end of a local socketpair."""

    def __init__(self, fail: bool = False):
        """Record whether opening a channel should fail."""
        self.fail = fail
        self.remote_ends: list[socket.socket] = []
        self.requests: list[tuple] = []

    def open_channel(self, kind, dest, origin):
        """Return a socket standing in for the forwarded channel, or raise if told to."""
        self.requests.append((kind, dest))
        if self.fail:
            raise OSError("channel refused")
        here, there = socket.socketpair()
        there.settimeout(5)
        self.remote_ends.append(there)
        return here


def _tunnel(monkeypatch, transport):
    """An _SSHTunnel whose SSH client is fake but whose local listener is real."""
    ssh = MagicMock()
    ssh.get_transport.return_value = transport
    monkeypatch.setattr(client.paramiko, "SSHClient", MagicMock(return_value=ssh))
    t = client._SSHTunnel("gw", 22, "u", "p", "127.0.0.1", 27017)
    return t, ssh


class TestSSHTunnel:
    def test_local_connection_is_forwarded_to_the_remote_port(self, monkeypatch):
        """Bytes written to the local listener arrive on a direct-tcpip channel to MongoDB."""
        transport = _FakeTransport()
        t, ssh = _tunnel(monkeypatch, transport)
        try:
            ssh.connect.assert_called_once_with(
                hostname="gw", port=22, username="u", password="p"
            )
            with socket.create_connection(("127.0.0.1", t.local_bind_port), 5) as c:
                c.sendall(b"hello")
                for _ in range(100):
                    if transport.remote_ends:
                        break
                    threading.Event().wait(0.05)
                assert transport.requests[0] == ("direct-tcpip", ("127.0.0.1", 27017))
                assert _recv_exact(transport.remote_ends[0], 5) == b"hello"
        finally:
            t.stop()
        ssh.close.assert_called_once()

    def test_refused_channel_closes_the_local_connection(self, monkeypatch):
        """If the remote side refuses the channel, the local client sees a closed socket
        rather than hanging forever."""
        t, _ = _tunnel(monkeypatch, _FakeTransport(fail=True))
        try:
            with socket.create_connection(("127.0.0.1", t.local_bind_port), 5) as c:
                c.settimeout(5)
                assert c.recv(1) == b""
        finally:
            t.stop()
