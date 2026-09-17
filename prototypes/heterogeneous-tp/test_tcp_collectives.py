import json
import socket
import struct
import unittest
from tcp_collectives import TCPCollectives, read_exact, MAX_HEADER

class Frames(unittest.TestCase):
    def setUp(self):
        self.left, self.right = socket.socketpair()
        self.transport = TCPCollectives.__new__(TCPCollectives)
        self.transport.sock = self.left
        self.transport.sequence = 1
    def tearDown(self):
        self.left.close(); self.right.close()
    def frame(self, **changes):
        header = dict(version=1, seq=1, op='sum', dtype='f32le', shape=[1], nbytes=4)
        header.update(changes)
        data = json.dumps(header).encode()
        self.right.sendall(struct.pack('!I', len(data)) + data + struct.pack('<f', 1.25))
    def test_wire_endianness(self):
        self.frame()
        raw = self.transport._recv('sum', 'f32le', [1])
        self.assertEqual(self.transport._tensor(raw, [1]).item(), 1.25)
    def test_wrong_sequence(self):
        self.frame(seq=2)
        with self.assertRaises(ValueError): self.transport._recv('sum', 'f32le', [1])
    def test_wrong_shape(self):
        self.frame(shape=[2])
        with self.assertRaises(ValueError): self.transport._recv('sum', 'f32le', [1])
    def test_wrong_length(self):
        self.frame(nbytes=0)
        with self.assertRaises(ValueError): self.transport._recv('sum', 'f32le', [1])
    def test_oversized_header(self):
        self.right.sendall(struct.pack('!I', MAX_HEADER+1))
        with self.assertRaises(ValueError): self.transport._recv('sum', 'f32le', [1])
    def test_truncated_frame(self):
        self.right.sendall(b'a'); self.right.close()
        with self.assertRaises(ConnectionError): read_exact(self.left, 2)

if __name__ == '__main__': unittest.main()
