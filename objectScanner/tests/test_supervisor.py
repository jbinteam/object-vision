"""Exercise cancellation without a camera, display, or CUDA."""
import contextlib
import io
import multiprocessing as mp
import signal
import time
import unittest

from supervisor import monitor


def stuck_worker(connection, stop):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    connection.send("Synthetic blocked native operation")
    while True:
        time.sleep(10)


def responsive_worker(connection, stop):
    connection.send("Ready")
    stop.wait(3)
    connection.close()


class InterruptOnce:
    def __init__(self, connection):
        self.connection = connection
        self.interrupted = False

    def poll(self, timeout):
        if not self.interrupted:
            self.interrupted = True
            raise KeyboardInterrupt
        return self.connection.poll(timeout)

    def recv(self):
        return self.connection.recv()

    def close(self):
        self.connection.close()


class SupervisorTests(unittest.TestCase):
    def worker(self, target):
        ctx = mp.get_context("spawn")
        reader, writer = ctx.Pipe(duplex=False)
        stop = ctx.Event()
        process = ctx.Process(target=target, args=(writer, stop))
        process.start()
        writer.close()
        self.addCleanup(self.cleanup, process)
        return process, reader, stop

    @staticmethod
    def cleanup(process):
        if process.is_alive():
            process.kill()
        process.join(2)

    def test_stuck_worker_is_killed_without_cooperation(self):
        process, reader, stop = self.worker(stuck_worker)
        started = time.monotonic()
        with contextlib.redirect_stderr(io.StringIO()) as error:
            code = monitor(process, reader, stop, timeout=0.3)
        self.assertEqual(code, 124)
        self.assertFalse(process.is_alive())
        self.assertLess(time.monotonic() - started, 6)
        self.assertIn("Synthetic blocked native operation", error.getvalue())

    def test_ctrl_c_requests_graceful_stop(self):
        process, reader, stop = self.worker(responsive_worker)
        with contextlib.redirect_stdout(io.StringIO()):
            code = monitor(process, InterruptOnce(reader), stop, timeout=5)
        self.assertTrue(stop.is_set())
        self.assertEqual(code, 0)
        self.assertFalse(process.is_alive())

    def test_ctrl_c_stops_unresponsive_worker(self):
        process, reader, stop = self.worker(stuck_worker)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = monitor(process, InterruptOnce(reader), stop, timeout=5, cancel_grace=0.3)
        self.assertEqual(code, 130)
        self.assertFalse(process.is_alive())
