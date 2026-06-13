#!/usr/bin/env python3
"""Probe / listen for YEE TS3-19 (XfdAp / LC329-style) WiFi video."""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import threading
import time
from pathlib import Path

HOST = "192.168.43.1"
OUT = Path(__file__).resolve().parent / "snapshots" / "yee_probe"

CTRL_8090 = bytes.fromhex("aa80800080008055")
START_8080 = bytes.fromhex("4276")
FLYLINK = bytes.fromhex("0155aa33")


def log(msg: str) -> None:
    print(msg, flush=True)


def tcp_scan(host: str) -> list[int]:
    open_ports: list[int] = []
    ports = list(range(1, 1025)) + [554, 8554, 8080, 8090, 8888, 9000, 9527, 10000, 17700, 32108]
    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.12)
        try:
            if s.connect_ex((host, port)) == 0:
                open_ports.append(port)
                log(f"OPEN TCP {port}")
        finally:
            s.close()
    return open_ports


def udp_listen(port: int, seconds: float, sink: list[tuple]) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", port))
    except OSError as exc:
        log(f"bind skip {port}: {exc}")
        s.close()
        return
    s.settimeout(0.4)
    end = time.time() + seconds
    while time.time() < end:
        try:
            data, addr = s.recvfrom(65535)
            sink.append((port, addr, data))
            log(f"UDP recv local:{port} from {addr[0]}:{addr[1]} len={len(data)} head={data[:16].hex()}")
        except socket.timeout:
            pass
    s.close()


def fire_starts(host: str, local_ip: str) -> None:
    ipb = bytes(int(x) for x in local_ip.split("."))
    payloads = [CTRL_8090, START_8080, FLYLINK, bytes.fromhex("ef000400"), b"DISCOVER", b"START", ipb + struct.pack(">H", 8080)]
    ports = [8090, 8080, 9527, 1008, 8800, 8899, 10210, 17900, 32108, 10000, 7777]
    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.15)
        for payload in payloads:
            try:
                s.sendto(payload, (host, port))
                data, addr = s.recvfrom(4096)
                log(f"UDP reply port {port} from {addr} len={len(data)} head={data[:16].hex()}")
            except OSError:
                pass
        s.close()


def save_chunks(chunks: list[tuple]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (_, addr, data) in enumerate(chunks, 1):
        path = OUT / f"pkt_{i}_{addr[0]}_{addr[1]}_{len(data)}.bin"
        path.write_bytes(data)
        if b"\xff\xd8\xff" in data:
            start = data.find(b"\xff\xd8\xff")
            end = data.find(b"\xff\xd9", start)
            if end > start:
                jpg = OUT / f"frame_{i}.jpg"
                jpg.write_bytes(data[start : end + 2])
                log(f"saved JPEG {jpg}")


def local_ip_for(host: str) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.settimeout(1)
        s.connect((host, 1))
        return s.getsockname()[0]
    except OSError:
        return "192.168.43.2"
    finally:
        s.close()


def cmd_probe(host: str) -> int:
    log(f"=== probe {host} ===")
    lip = local_ip_for(host)
    log(f"local ip: {lip}")
    open_ports = tcp_scan(host)
    log(f"TCP summary: {open_ports or 'none'}")
    fire_starts(host, lip)
    log("probe done")
    return 0


def cmd_listen(host: str, seconds: float) -> int:
    log(f"=== listen {seconds:.0f}s ===")
    log("Start Cam802 on phone while this runs.")
    chunks: list[tuple] = []
    ports = [8080, 8090, 8888, 9527, 10000, 10008, 10210, 17700, 32108, 49152, 5000, 7777]
    threads = [threading.Thread(target=udp_listen, args=(p, seconds, chunks), daemon=True) for p in ports]
    for t in threads:
        t.start()
    time.sleep(0.3)
    lip = local_ip_for(host)
    end = time.time() + seconds
    while time.time() < end:
        fire_starts(host, lip)
        time.sleep(2)
    for t in threads:
        t.join()
    log(f"packets captured: {len(chunks)}")
    if chunks:
        save_chunks(chunks)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["probe", "listen"])
    p.add_argument("--host", default=HOST)
    p.add_argument("--seconds", type=float, default=45.0)
    args = p.parse_args()
    if args.command == "probe":
        return cmd_probe(args.host)
    return cmd_listen(args.host, args.seconds)


if __name__ == "__main__":
    sys.exit(main())