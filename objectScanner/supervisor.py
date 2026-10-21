"""Keep the CLI responsive when a native camera/Open3D call does not return."""
import faulthandler
import logging
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time


class ProgressHandler(logging.Handler):
    def __init__(self, connection):
        super().__init__()
        self.connection = connection

    def emit(self, record):
        self.connection.send(record.getMessage())


def _scan_worker(args, connection, stop):
    # The parent handles terminal Ctrl+C, then requests graceful shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.safe:
        # Avoid nested BLAS/OpenMP thread pools in the conservative mode.
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = "1"
    logger = logging.getLogger("object_scanner")
    handler = ProgressHandler(connection)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    trace_path = args.output.with_name(args.output.stem + "_native.log")
    try:
        with trace_path.open("w") as trace:
            faulthandler.enable(file=trace)
            if hasattr(signal, "SIGUSR1"):
                faulthandler.register(signal.SIGUSR1, file=trace, all_threads=True)
            try:
                from scanner import scan
                scan(args, stop=stop)
            finally:
                if hasattr(signal, "SIGUSR1"):
                    faulthandler.unregister(signal.SIGUSR1)
                faulthandler.disable()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    finally:
        logger.removeHandler(handler)
        connection.close()


def monitor(process, connection, stop, timeout, cancel_grace=3.0):
    """Return child exit status, 124 on stalled work, or 130 on cancellation."""
    last_progress = time.monotonic()
    stage = "Loading camera/reconstruction libraries"
    cancelled_at = None
    received = False
    status = None
    try:
        while process.is_alive():
            try:
                if connection.poll(0.1):
                    try:
                        stage = connection.recv()
                        last_progress = time.monotonic()
                        received = True
                    except EOFError:
                        process.join(0.1)
                now = time.monotonic()
                if cancelled_at is not None and now - cancelled_at >= cancel_grace:
                    status = 130
                    print("Worker did not stop after Ctrl+C; terminating it. The in-memory scan may be lost.", file=sys.stderr, flush=True)
                    break
                # Cold imports can be slower than individual processing calls.
                limit = timeout if received else max(timeout, 30)
                if now - last_progress > limit:
                    status = 124
                    print(f"No progress for {limit:g}s at: {stage}. Stopping worker; the in-memory scan is not saved.", file=sys.stderr, flush=True)
                    break
            except KeyboardInterrupt:
                if cancelled_at is not None:
                    status = 130
                    break
                cancelled_at = time.monotonic()
                stop.set()
                print("Stop requested; allowing up to 3 seconds to save and exit. Ctrl+C again stops immediately.", flush=True)
    finally:
        if process.is_alive():
            # Dump the native-call Python stack before terminating the worker.
            if hasattr(signal, "SIGUSR1") and received:
                try:
                    os.kill(process.pid, signal.SIGUSR1)
                    process.join(0.15)
                except ProcessLookupError:
                    pass
            process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join(1)
        else:
            process.join()
        connection.close()
    return status if status is not None else (process.exitcode or 0)


def run_scan(args):
    from scanner import output_paths
    # Resolve and protect the output path before creating a native trace file.
    paths = output_paths(args)
    args.output = paths["cloud"]
    native_path = args.output.with_name(args.output.stem + "_native.log")
    if native_path.exists():
        raise RuntimeError(f"Output already exists: {native_path}")
    context = mp.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    stop = context.Event()
    worker = context.Process(target=_scan_worker, args=(args, writer, stop))
    worker.start()
    writer.close()
    print(f"Watchdog active: {args.stall_timeout:g}s per operation. Log: {paths['log']}", flush=True)
    return monitor(worker, reader, stop, args.stall_timeout)
