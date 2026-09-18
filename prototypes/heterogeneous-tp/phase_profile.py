"""Opt-in per-rank diagnostics, never a measurement of pure network latency."""
from contextlib import contextmanager
from time import perf_counter

from tcp_collectives import TCPCollectives


class PhaseProfile:
    def __init__(self, synchronize):
        self.synchronize = synchronize
        self.active = False
        self.phase = "prefill"
        self.stats = {}

    def start(self, enabled):
        self.stats = {}
        self.active = enabled

    def finish(self):
        self.active = False
        return self.stats

    @contextmanager
    def measure(self, name, synchronize=False, payload_bytes=0):
        if not self.active:
            yield
            return
        start = perf_counter()
        try:
            yield
            if synchronize:
                self.synchronize()
        finally:
            elapsed = perf_counter() - start
            stat = self.stats.setdefault(self.phase, {}).setdefault(
                name, {"calls": 0, "elapsed_s": 0.0, "payload_bytes": 0})
            stat["calls"] += 1
            stat["elapsed_s"] += elapsed
            stat["payload_bytes"] += payload_bytes


class ProfiledTCPCollectives(TCPCollectives):
    def __init__(self, profile, **kwargs):
        self.profile = profile
        super().__init__(**kwargs)

    def _send(self, op, dtype, shape, payload):
        with self.profile.measure("socket_send", payload_bytes=len(payload)):
            return super()._send(op, dtype, shape, payload)

    def _recv(self, op, dtype, shape):
        with self.profile.measure("socket_receive"):
            payload = super()._recv(op, dtype, shape)
        if self.profile.active:
            self.profile.stats[self.profile.phase]["socket_receive"]["payload_bytes"] += len(payload)
        return payload
