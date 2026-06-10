#!/usr/bin/env python3
"""
AGM Taipan Stream Explorer
Capture and analyze network traffic while the AGM Connect app streams live view.

Usage:
  1. Connect laptop to Taipan WiFi hotspot (same network as phone).
  2. Start live view in AGM Connect on your phone.
  3. Run: python agm_taipan_stream_explorer.py
  4. Optional: python agm_taipan_stream_explorer.py --probe-rtsp
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from collections import defaultdict
from typing import Optional

try:
    from scapy.all import IP, TCP, UDP, Raw, conf, get_if_list, sniff
    from scapy.arch.windows import get_windows_if_list
except ImportError:
    print("Missing scapy. Run: pip install scapy")
    sys.exit(1)


def npcap_available() -> bool:
    if sys.platform != "win32":
        return True
    try:
        return bool(conf.use_pcap and get_if_list())
    except Exception:
        return False

# Common video codec / container signatures in raw payloads
VIDEO_SIGNATURES = {
    b"\x00\x00\x00\x01": "H.264/HEVC NAL start code",
    b"\x00\x00\x01": "H.264/HEVC NAL start code (3-byte)",
    b"\xff\xd8\xff": "MJPEG/JPEG",
    b"ftyp": "MP4/MOV fragment",
    b"RIFF": "AVI/WebM container",
    b"\x47": "MPEG-TS sync byte (possible)",
    b"RTSP": "RTSP text",
    b"DESCRIBE": "RTSP DESCRIBE",
    b"OPTIONS rtsp": "RTSP OPTIONS",
}

COMMON_RTSP_PATHS = [
    "/live",
    "/live.sdp",
    "/stream",
    "/stream1",
    "/h264",
    "/video",
    "/cam/realmonitor",
    "/11",
    "/0",
    "/1",
]

print("=== AGM Taipan Stream Explorer ===")
print("1. Turn on Taipan WiFi hotspot")
print("2. Connect your PHONE to the hotspot")
print("3. Open AGM Connect app and start live view")
print("4. Run this script on your laptop connected to the SAME hotspot\n")


def get_taipan_interface() -> Optional[str]:
    """Pick the WiFi interface on the Taipan hotspot subnet (10.15.12.x)."""
    candidates = []
    for iface in get_windows_if_list():
        ips = iface.get("ips") or []
        ipv4 = [ip for ip in ips if "." in ip and not ip.startswith("169.254")]
        on_taipan = any(ip.startswith("10.15.12.") for ip in ipv4)
        if on_taipan:
            candidates.append(
                {
                    "name": iface.get("name", ""),
                    "guid": iface.get("guid", ""),
                    "ips": ipv4,
                }
            )

    if not candidates:
        return None

    # Prefer interface literally named Wi-Fi; guid already includes {braces}
    for c in candidates:
        if c["name"] == "Wi-Fi":
            return c["name"]

    return candidates[0]["name"]


def list_interfaces() -> None:
    print("Available interfaces:")
    for iface in get_windows_if_list():
        ips = ", ".join(iface.get("ips") or []) or "(no IP)"
        print(f"  {iface.get('name', '?'):30} {ips}")


def detect_video_signature(payload: bytes) -> list[str]:
    hits = []
    for sig, label in VIDEO_SIGNATURES.items():
        if sig in payload[:512]:
            hits.append(label)
    return hits


def packet_callback_factory(stats: dict, verbose: bool):
    def packet_callback(packet):
        if IP not in packet:
            return

        src = packet[IP].src
        dst = packet[IP].dst
        proto = "?"
        sport = dport = 0
        plen = len(packet)

        if UDP in packet:
            proto = "UDP"
            sport = packet[UDP].sport
            dport = packet[UDP].dport
        elif TCP in packet:
            proto = "TCP"
            sport = packet[TCP].sport
            dport = packet[TCP].dport
        else:
            return

        key = (proto, src, sport, dst, dport)
        stats["flows"][key]["bytes"] += plen
        stats["flows"][key]["pkts"] += 1

        payload = b""
        if Raw in packet:
            payload = bytes(packet[Raw].load)
            sigs = detect_video_signature(payload)
            if sigs:
                stats["video_hits"].append(
                    {
                        "time": time.strftime("%H:%M:%S"),
                        "flow": f"{src}:{sport} -> {dst}:{dport} ({proto})",
                        "signatures": sigs,
                        "sample": payload[:32].hex(),
                    }
                )

        if verbose and plen > 200:
            print(
                f"[{time.strftime('%H:%M:%S')}] {proto} {src}:{sport} -> {dst}:{dport} | Len: {plen}"
            )
            if payload and detect_video_signature(payload):
                print(f"    ^ video signature: {detect_video_signature(payload)}")

    return packet_callback


def print_summary(stats: dict, top_n: int = 15) -> None:
    flows = stats["flows"]
    if not flows:
        print("\nNo UDP/TCP packets captured.")
        return

    ranked = sorted(flows.items(), key=lambda kv: kv[1]["bytes"], reverse=True)

    print("\n=== Top bandwidth flows (likely video candidates) ===")
    print(f"{'Proto':<5} {'Source':<22} {'Destination':<22} {'Pkts':>6} {'Bytes':>10} {'KB/s':>8}")
    print("-" * 80)

    duration = max(stats.get("duration", 1), 1)
    for (proto, src, sport, dst, dport), data in ranked[:top_n]:
        kbps = (data["bytes"] / 1024) / duration
        print(
            f"{proto:<5} {src}:{sport:<16} {dst}:{dport:<16} "
            f"{data['pkts']:>6} {data['bytes']:>10} {kbps:>8.1f}"
        )

    if stats["video_hits"]:
        print("\n=== Video stream signatures detected ===")
        seen = set()
        for hit in stats["video_hits"]:
            key = (hit["flow"], tuple(hit["signatures"]))
            if key in seen:
                continue
            seen.add(key)
            print(f"  [{hit['time']}] {hit['flow']}")
            print(f"    Signatures: {', '.join(hit['signatures'])}")
            print(f"    Sample: {hit['sample']}")

    # Suggest probe targets from high-bandwidth flows
    print("\n=== Suggested probe targets ===")
    for (proto, src, sport, dst, dport), data in ranked[:5]:
        if data["bytes"] < 5000:
            continue
        # Stream usually flows TO the phone; server is often the non-phone side
        for host, port in ((dst, dport), (src, sport)):
            if host.startswith("10.15.12."):
                print(f"  {proto} {host}:{port}  ({data['bytes']} bytes)")


def probe_rtsp(hosts: list[str], ports: list[int], timeout: float = 2.0) -> None:
    print("\n=== RTSP URL probe ===")
    for host in hosts:
        for port in ports:
            for path in COMMON_RTSP_PATHS:
                url = f"rtsp://{host}:{port}{path}"
                try:
                    sock = socket.create_connection((host, port), timeout=timeout)
                    req = (
                        f"OPTIONS {path} RTSP/1.0\r\n"
                        f"CSeq: 1\r\n"
                        f"User-Agent: AGM-Taipan-Explorer\r\n\r\n"
                    )
                    sock.sendall(req.encode())
                    sock.settimeout(timeout)
                    resp = sock.recv(4096)
                    sock.close()
                    if resp:
                        text = resp.decode("utf-8", errors="replace").split("\r\n")[0]
                        print(f"  [HIT] {url} -> {text}")
                except (OSError, socket.timeout):
                    pass


def probe_tcp_ports(host: str, ports: list[int], timeout: float = 1.5) -> None:
    print(f"\n=== TCP port probe on {host} ===")
    for port in ports:
        try:
            sock = socket.create_connection((host, port), timeout=timeout)
            sock.sendall(b"\x00")
            sock.settimeout(timeout)
            try:
                resp = sock.recv(256)
            except socket.timeout:
                resp = b""
            sock.close()
            status = f"open, {len(resp)} byte response" if resp else "open, no response"
            print(f"  Port {port}: {status}")
            if resp:
                print(f"    hex: {resp[:64].hex()}")
        except (OSError, socket.timeout):
            pass


def run_capture(iface: str, timeout: int, verbose: bool) -> dict:
    stats = {
        "flows": defaultdict(lambda: {"bytes": 0, "pkts": 0}),
        "video_hits": [],
        "duration": timeout,
    }
    cb = packet_callback_factory(stats, verbose)

    print(f"Capturing on: {iface}")
    print(f"Duration: {timeout}s — start live view on phone now (Ctrl+C to stop early)\n")

    start = time.time()
    try:
        sniff(iface=iface, prn=cb, timeout=timeout, store=False)
    except KeyboardInterrupt:
        print("\nCapture stopped by user.")
    except OSError as e:
        print(f"\nCapture failed: {e}")
        print("\nOn Windows you need Npcap installed (https://npcap.com/)")
        print("Install with 'WinPcap API-compatible Mode' enabled, then re-run as Administrator.")
        sys.exit(1)

    stats["duration"] = max(time.time() - start, 1)
    return stats


def main():
    parser = argparse.ArgumentParser(description="AGM Taipan live stream network explorer")
    parser.add_argument("--iface", help="Scapy interface (auto-detected if omitted)")
    parser.add_argument("--timeout", type=int, default=60, help="Capture seconds (default: 60)")
    parser.add_argument("--verbose", action="store_true", help="Print large packets live")
    parser.add_argument("--list-ifaces", action="store_true", help="List interfaces and exit")
    parser.add_argument("--probe-rtsp", action="store_true", help="Probe common RTSP URLs on hotspot")
    parser.add_argument("--probe-only", action="store_true", help="Skip capture; only run RTSP/TCP probes")
    parser.add_argument("--probe-host", default="10.15.12.1", help="Host for RTSP/TCP probes")
    args = parser.parse_args()

    if args.list_ifaces:
        list_interfaces()
        if not npcap_available():
            print("\nNote: Npcap not detected — install from https://npcap.com/ for packet capture.")
        return

    if not npcap_available() and not args.probe_only:
        print("Npcap is required for packet capture on Windows.")
        print("  1. Download: https://npcap.com/#download")
        print("  2. Install with 'Install Npcap in WinPcap API-compatible Mode' checked")
        print("  3. Re-run this script as Administrator")
        print("\nYou can still probe RTSP without capture: --probe-only --probe-rtsp")
        sys.exit(1)

    stats = {"flows": defaultdict(lambda: {"bytes": 0, "pkts": 0}), "video_hits": [], "duration": 1}

    if not args.probe_only:
        iface = args.iface or get_taipan_interface()
        if not iface:
            print("Could not auto-detect Taipan WiFi (expected 10.15.12.x).")
            list_interfaces()
            print("\nConnect to the Taipan hotspot, or pass --iface manually.")
            sys.exit(1)
        stats = run_capture(iface, args.timeout, args.verbose)
        print_summary(stats)

    if args.probe_rtsp or args.probe_only:
        # Gateway is often the scope; also try .1 and any high-traffic peers
        hosts = {args.probe_host, "10.15.12.1"}
        for (proto, src, sport, dst, dport), data in sorted(
            stats["flows"].items(), key=lambda kv: kv[1]["bytes"], reverse=True
        )[:5]:
            if data["bytes"] > 5000:
                hosts.add(dst)
                hosts.add(src)
        probe_rtsp(sorted(hosts), [554, 8554, 10554, 7070, 8080])
        hot_ports = sorted(
            {dport for (_, _, _, _, dport), d in stats["flows"].items() if d["bytes"] > 10000}
            | {sport for (_, _, sport, _, _), d in stats["flows"].items() if d["bytes"] > 10000}
        )
        if hot_ports:
            probe_tcp_ports(args.probe_host, hot_ports[:10])


if __name__ == "__main__":
    main()
