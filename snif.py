#!/usr/bin/env python3
# snif.py — Sniffer + API de eventos para Open5GS Graph

import json
import subprocess
import threading
import time
from datetime import datetime
import re

from flask import Flask, jsonify, request
from scapy.all import sniff, IFACES, Raw
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import ARP

# ================================================================
# CONFIGURACIÓN
# ================================================================

SERVICE_BY_IP = {
    # loopback Open5GS
    "127.0.0.2":  "MME",
    "127.0.0.3":  "SGWC",
    "127.0.0.4":  "SMF",
    "127.0.0.5":  "AMF",
    "127.0.0.6":  "SGWU",
    "127.0.0.7":  "UPF",
    "127.0.0.8":  "HSS",
    "127.0.0.9":  "PCRF",
    "127.0.0.10": "NRF",
    "127.0.0.11": "AUSF",
    "127.0.0.12": "UDM",
    "127.0.0.13": "PCF",
    "127.0.0.14": "NSSF",
    "127.0.0.15": "BSF",
    "127.0.0.20": "UDR",
    "127.0.0.22": "SCP",

    # acceso radio
    "10.45.1.1": "GNB",
    "10.45.2.1": "GNB2",
    "10.45.1.2": "UE1",
    "10.45.2.2": "UE2",
    # interfaz N2 gNB ↔ AMF
    "10.53.1.1": "GNB",
    "10.53.1.11": "GNB2",
    "10.53.1.2": "AMF",
}

# Procesos Open5GS a vigilar para estado UP/DOWN
OPEN5GS_PROCESSES = {
    "NRF":  "open5gs-nrfd",
    "SCP":  "open5gs-scpd",
    "UPF":  "open5gs-upfd",
    "SMF":  "open5gs-smfd",
    "AMF":  "open5gs-amfd",
    "AUSF": "open5gs-ausfd",
    "UDM":  "open5gs-udmd",
    "PCF":  "open5gs-pcfd",
    "NSSF": "open5gs-nssfd",
    "BSF":  "open5gs-bsfd",
    "UDR":  "open5gs-udrd",
}

MAX_EVENTS = 1500

# ================================================================
# UTILIDADES
# ================================================================

def label_ip(ip: str) -> str:
    """Devuelve NOMBRE(IP) si existe mapeo, si no solo IP."""
    name = SERVICE_BY_IP.get(ip)
    return f"{name}({ip})" if name else ip


def tcp_flags_list(tcp):
    f = int(tcp.flags)
    out = []
    if f & 0x02: out.append("SYN")
    if f & 0x10: out.append("ACK")
    if f & 0x01: out.append("FIN")
    if f & 0x04: out.append("RST")
    if f & 0x08: out.append("PSH")
    return out


HTTP_METHOD_RE = re.compile(r"(GET|POST|PUT|PATCH|DELETE)\s+(/[^ ]*)")


def infer_nf_from_http(payload: bytes):
    """
    Detecta si el destino es NRF, UDM, AUSF, PCF, etc.
    según el path HTTP SBI.
    """
    try:
        text = payload.decode("utf-8", errors='ignore')
    except:
        return None

    m = HTTP_METHOD_RE.search(text)
    if not m:
        return None

    path = m.group(2)

    mapping = [
        ("nudm-",  "UDM"),
        ("nnrf-",  "NRF"),
        ("nrf-",   "NRF"),
        ("npcf-",  "PCF"),
        ("namf-",  "AMF"),
        ("nausf-", "AUSF"),
        ("nnssf-", "NSSF"),
        ("nbsf-",  "BSF"),
        ("nudr-",  "UDR"),
    ]

    for prefix, name in mapping:
        if prefix in path:
            return name

    return None


def try_extract_json(payload):
    start = payload.find(b"{")
    end = payload.rfind(b"}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(payload[start:end+1].decode("utf-8", errors="ignore"))
    except:
        return None


def get_service_status():
    """Retorna dict {NF: True/False} usando ps aux."""
    try:
        ps = subprocess.check_output(["ps", "aux"], text=True)
    except:
        return {}

    st = {}
    for nf, proc in OPEN5GS_PROCESSES.items():
        st[nf] = (proc in ps)
    return st

# ================================================================
# API HTTP
# ================================================================

app = Flask(__name__)
EVENTS = []

def log_event(line: str):
    EVENTS.append(line)
    if len(EVENTS) > MAX_EVENTS:
        del EVENTS[0]


@app.route("/events")
def get_events():
    try:
        limit = int(request.args.get("limit", "200"))
    except:
        limit = 200
    limit = min(max(limit, 1), MAX_EVENTS)
    return jsonify(EVENTS[-limit:])


@app.route("/events/clear", methods=["POST"])
def clear_events():
    EVENTS.clear()
    return jsonify({"status": "ok"})

# ================================================================
# SNIFFER
# ================================================================

def handle(pkt):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    # filtramos solo IP/ARP
    if IP not in pkt and ARP not in pkt:
        return

    # ARP
    if ARP in pkt:
        arp = pkt[ARP]
        log_event(f"{ts} ARP {label_ip(arp.psrc)} -> {label_ip(arp.pdst)} op={arp.op}")
        return

    ip = pkt[IP]
    if ip.src == ip.dst:
        return

    src_label = label_ip(ip.src)
    dst_label = label_ip(ip.dst)

    # ---------------- TCP ----------------
    if TCP in pkt:
        tcp = pkt[TCP]
        raw = pkt[Raw].load if Raw in pkt and pkt[Raw].load else None

        # detectar NF por HTTP SBI
        if raw:
            nf = infer_nf_from_http(raw)
            if nf:
                if SERVICE_BY_IP.get(ip.src) and not SERVICE_BY_IP.get(ip.dst):
                    dst_label = f"{nf}({ip.dst})"
                elif SERVICE_BY_IP.get(ip.dst) and not SERVICE_BY_IP.get(ip.src):
                    src_label = f"{nf}({ip.src})"

        base = f"{ts}  {src_label} -> {dst_label}  len={len(pkt)}"
        flags = tcp_flags_list(tcp)
        log_event(f"{base}  TCP {tcp.sport}->{tcp.dport} flags={flags}")

        if raw:
            js = try_extract_json(raw)
            if js:
                log_event(f"    JSON: {js}")
            else:
                log_event(f"    payload {raw[:60]!r}")
        return

    # ---------------- UDP ----------------
    if UDP in pkt:
        udp = pkt[UDP]
        base = f"{ts}  {src_label} -> {dst_label}  len={len(pkt)}"
        log_event(f"{base}  UDP {udp.sport}->{udp.dport}")
        return

    # ---------------- ICMP ----------------
    if ICMP in pkt:
        ic = pkt[ICMP]
        base = f"{ts}  {src_label} -> {dst_label} len={len(pkt)}"
        log_event(f"{base} ICMP type={ic.type} code={ic.code}")
        return

    # ---------------- OTROS ----------------
    base = f"{ts}  {src_label} -> {dst_label} len={len(pkt)}"
    log_event(f"{base} PROTO={ip.proto}")


def get_interfaces_all():
    return [iface.name for iface in IFACES.values()] or ["any"]


def sniffer_thread():
    sniff(iface=get_interfaces_all(), prn=handle, store=False)

# ================================================================
# LOOP ESTADO DE SERVICIOS
# ================================================================

def service_status_loop():
    while True:
        st = get_service_status()
        parts = [f"{k}={'UP' if v else 'DOWN'}" for k, v in st.items()]
        log_event("[SERVICE] " + " ".join(parts))
        time.sleep(2)

# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("Iniciando sniffer en todas las interfaces...")

    threading.Thread(target=sniffer_thread, daemon=True).start()
    threading.Thread(target=service_status_loop, daemon=True).start()

    app.run(host="0.0.0.0", port=8000)
