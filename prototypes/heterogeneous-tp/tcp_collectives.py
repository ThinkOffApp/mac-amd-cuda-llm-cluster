"""Two-rank CPU-staged correctness transport; trusted private networks only.

Wire format: big-endian JSON-header length, UTF-8 header, little-endian f32 body.
No pickle, native structs, device pointers, or assumptions about host endianness.
This deliberately simple root reduction is not an optimized inference transport.
"""
import json
import os
import socket
import struct
import time

import numpy as np
import torch

MAX_HEADER = 4096
MAX_PAYLOAD = 64 * 1024 * 1024


def read_exact(sock, size):
    chunks = bytearray()
    while len(chunks) < size:
        part = sock.recv(size - len(chunks))
        if not part:
            raise ConnectionError('Peer closed before frame completed')
        chunks.extend(part)
    return bytes(chunks)


class TCPCollectives:
    def __init__(self, rank=None, host=None, port=None, timeout=60):
        self.rank = int(os.environ.get('RANK', '0')) if rank is None else rank
        self.world_size = int(os.environ.get('WORLD_SIZE', '2'))
        if self.rank not in (0, 1) or self.world_size != 2:
            raise ValueError('TCP correctness backend requires exactly two ranks')
        host = host or os.environ.get('MASTER_ADDR', '127.0.0.1')
        port = int(port or os.environ.get('MASTER_PORT', '29517'))
        self.sequence = 0
        if self.rank == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.settimeout(timeout)
                listener.bind((host, port))
                listener.listen(1)
                self.sock, _ = listener.accept()
        else:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    self.sock = socket.create_connection((host, port), timeout=max(.1, deadline-time.monotonic()))
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.1)
        self.sock.settimeout(timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            if self.rank == 0:
                self._send('hello', 'json', [], b'0')
                body = self._recv('hello', 'json', [])
                if body != b'1': raise ValueError('Expected peer rank 1')
            else:
                body = self._recv('hello', 'json', [])
                if body != b'0': raise ValueError('Expected peer rank 0')
                self._send('hello', 'json', [], b'1')
            self.sequence += 1
        except BaseException:
            self.close()
            raise

    def _send(self, op, dtype, shape, payload):
        header = json.dumps(dict(version=1, seq=self.sequence, op=op,
                                 dtype=dtype, shape=shape, nbytes=len(payload))).encode()
        if len(header) > MAX_HEADER or len(payload) > MAX_PAYLOAD:
            raise ValueError('Frame exceeds transport limit')
        self.sock.sendall(struct.pack('!I', len(header)) + header + payload)

    def _recv(self, op, dtype, shape):
        length, = struct.unpack('!I', read_exact(self.sock, 4))
        if not 0 < length <= MAX_HEADER:
            raise ValueError('Invalid header size')
        header = json.loads(read_exact(self.sock, length))
        if not isinstance(header, dict): raise ValueError('Invalid frame header')
        expected = dict(version=1, seq=self.sequence, op=op, dtype=dtype, shape=shape)
        if any(header.get(k) != v for k, v in expected.items()):
            raise ValueError('Collective protocol, sequence, dtype or shape mismatch')
        size = header.get('nbytes')
        if type(size) is not int or not 0 <= size <= MAX_PAYLOAD:
            raise ValueError('Invalid payload size')
        if dtype == 'f32le':
            elements = 1
            for dim in shape: elements *= dim
            if size != elements * 4: raise ValueError('Tensor payload size mismatch')
        return read_exact(self.sock, size)

    @staticmethod
    def _check(tensor):
        if tensor.device.type != 'cpu' or tensor.dtype != torch.float32:
            raise ValueError('Transport accepts CPU float32 tensors only')
        if tensor.numel() * 4 > MAX_PAYLOAD: raise ValueError('Tensor exceeds limit')

    @staticmethod
    def _bytes(tensor):
        return tensor.contiguous().numpy().astype('<f4', copy=False).tobytes()

    @staticmethod
    def _tensor(payload, shape):
        return torch.from_numpy(np.frombuffer(payload, dtype='<f4').astype(np.float32, copy=True).reshape(shape))

    def broadcast(self, tensor, src=0):
        if src != 0: raise ValueError('Only root 0 supported')
        self._check(tensor)
        shape = list(tensor.shape)
        if self.rank == 0:
            self._send('broadcast', 'f32le', shape, self._bytes(tensor))
        else:
            tensor.copy_(self._tensor(self._recv('broadcast', 'f32le', shape), shape))
        self.sequence += 1

    def all_reduce(self, tensor, op=None):
        self._check(tensor)
        if op is not None and op != torch.distributed.ReduceOp.SUM:
            raise ValueError('Only SUM supported')
        shape = list(tensor.shape)
        if self.rank == 1:
            self._send('sum', 'f32le', shape, self._bytes(tensor))
            tensor.copy_(self._tensor(self._recv('sum-result', 'f32le', shape), shape))
        else:
            tensor.add_(self._tensor(self._recv('sum', 'f32le', shape), shape))
            self._send('sum-result', 'f32le', shape, self._bytes(tensor))
        self.sequence += 1

    def all_gather_object(self, reports, report):
        # JSON only; intentionally narrower than torch's arbitrary object API.
        body = json.dumps(report).encode()
        if self.rank == 1:
            self._send('reports', 'json', [], body)
            remote = self._recv('reports', 'json', [])
        else:
            remote = self._recv('reports', 'json', [])
            self._send('reports', 'json', [], body)
        reports[self.rank] = report
        reports[1-self.rank] = json.loads(remote)
        self.sequence += 1

    def get_rank(self): return self.rank
    def get_world_size(self): return self.world_size
    def destroy_process_group(self): self.close()
    def close(self): self.sock.close()
