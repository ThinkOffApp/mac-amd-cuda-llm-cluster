#!/usr/bin/env python3
"""Spark CUDA producer -> Mac Metal consumer, stock llama.cpp N-1 slot handoff."""
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path("/Users/petrus/split-test/spark-mac-handoff")
MAC_SLOTS = ROOT / "mac-slots"
LOGS = ROOT / "logs"
MAC_SLOTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519_agent_qcd")
SPARK_HOST = "10.10.10.2"
SPARK_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-i", SSH_KEY, f"petrus@{SPARK_HOST}"]
SCP = ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-i", SSH_KEY]

MAC_BIN = "/Users/petrus/llama.cpp/build/bin/llama-server"
MAC_MODEL = "/Volumes/t705/ModelArchive/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf"
SPARK_BIN = "/home/petrus/llama.cpp/build-rpc/bin/llama-server"
SPARK_MODEL = "/home/petrus/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf"
SPARK_SLOTS = "/tmp/spark-mac-handoff/slots"
SPARK_LOG = "/tmp/spark-mac-handoff/server.log"

SPARK_URL = "http://10.10.10.2:19083"
MAC_URL = "http://127.0.0.1:19084"
STATE_NAME = "before-final-prompt-token.bin"

PROMPT = "A careful experiment records the prompt, the cache state, and every generated token. " * 400


def req(base, path, body=None, timeout=180):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if data else {}
    t0 = time.time()
    with urllib.request.urlopen(
        urllib.request.Request(base + path, data=data, headers=headers), timeout=timeout
    ) as r:
        answer = json.load(r)
    return answer, time.time() - t0


def wait_health(base, timeout=240):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception as e:
            last = str(e)
        time.sleep(1)
    raise RuntimeError(f"health timeout {base}: {last}")


def ssh(cmd, timeout=30):
    return subprocess.check_output(SPARK_SSH + [cmd], text=True, timeout=timeout)


def start_spark():
    starter = "/tmp/spark-mac-handoff/start.sh"
    remote = f"""
mkdir -p {SPARK_SLOTS} /tmp/spark-mac-handoff
cat > {starter} << 'EOS'
#!/bin/bash
exec {SPARK_BIN} \\
  -m {SPARK_MODEL} -ngl 999 -c 4096 -np 1 -b 512 -ub 512 -t 4 \\
  --host 10.10.10.2 --port 19083 --slot-save-path {SPARK_SLOTS}/ \\
  --no-webui --log-verbosity 4 --cache-ram 0
EOS
chmod +x {starter}
nohup {starter} > {SPARK_LOG} 2>&1 &
echo $!
"""
    pid = ssh(remote).strip().splitlines()[-1]
    return int(pid)


def stop_spark(pid):
    try:
        ssh(f"kill {pid} 2>/dev/null || true")
    except Exception:
        pass


def main():
    report = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    mac_proc = None
    spark_pid = None
    mac_log = (LOGS / "mac-server.log").open("w")
    try:
        spark_pid = start_spark()
        report["spark_pid"] = spark_pid
        mac_cmd = [
            MAC_BIN, "-m", MAC_MODEL, "-ngl", "999", "-c", "4096", "-np", "1",
            "-b", "512", "-ub", "512", "-t", "4",
            "--host", "127.0.0.1", "--port", "19084",
            "--slot-save-path", str(MAC_SLOTS) + "/",
            "--no-webui", "--log-verbosity", "4", "--cache-ram", "0",
        ]
        mac_proc = subprocess.Popen(mac_cmd, stdout=mac_log, stderr=mac_log)
        report["mac_pid"] = mac_proc.pid
        report["spark_health_s"] = wait_health(SPARK_URL)
        report["mac_health_s"] = wait_health(MAC_URL)

        toks, tok_s = req(SPARK_URL, "/tokenize", {"content": PROMPT, "add_special": True})
        ids = toks["tokens"][:2112]
        assert len(ids) == 2112, len(ids)
        report["tokenize_s"] = tok_s
        report["n_prompt"] = len(ids)

        producer, prod_s = req(
            SPARK_URL,
            "/completion",
            {
                "prompt": ids[:-1],
                "id_slot": 0,
                "cache_prompt": True,
                "n_predict": 1,
                "temperature": 0,
                "seed": 42,
                "return_tokens": True,
            },
            timeout=180,
        )
        report["producer_wall_s"] = prod_s
        report["producer_timings"] = producer.get("timings")
        report["producer_discarded_token"] = producer.get("tokens")

        saved, save_s = req(SPARK_URL, "/slots/0?action=save", {"filename": STATE_NAME})
        report["save_wall_s"] = save_s
        report["save"] = saved

        remote = f"petrus@{SPARK_HOST}:{SPARK_SLOTS}/{STATE_NAME}"
        local = MAC_SLOTS / STATE_NAME
        t0 = time.time()
        subprocess.check_call(SCP + [remote, str(local)])
        copy_s = time.time() - t0
        size = local.stat().st_size
        with local.open("rb") as f:
            sha = hashlib.file_digest(f, "sha256").hexdigest()
        remote_sha = ssh(f"sha256sum {SPARK_SLOTS}/{STATE_NAME} | awk '{{print $1}}'").strip()
        report["copy_s"] = copy_s
        report["state_bytes"] = size
        report["state_sha256"] = sha
        report["spark_state_sha256"] = remote_sha
        report["copy_MBps"] = (size / copy_s / 1e6) if copy_s else None

        restored, restore_s = req(MAC_URL, "/slots/0?action=restore", {"filename": STATE_NAME})
        report["restore_wall_s"] = restore_s
        report["restore"] = restored

        handoff, handoff_s = req(
            MAC_URL,
            "/completion",
            {
                "prompt": ids,
                "id_slot": 0,
                "cache_prompt": True,
                "n_predict": 16,
                "temperature": 0,
                "seed": 42,
                "return_tokens": True,
            },
            timeout=180,
        )
        report["handoff_wall_s"] = handoff_s
        report["handoff_timings"] = handoff.get("timings")
        report["handoff_tokens"] = handoff.get("tokens")
        report["handoff_content"] = handoff.get("content")

        req(MAC_URL, "/slots/0?action=erase", {})
        native, native_s = req(
            MAC_URL,
            "/completion",
            {
                "prompt": ids,
                "id_slot": 0,
                "cache_prompt": True,
                "n_predict": 16,
                "temperature": 0,
                "seed": 42,
                "return_tokens": True,
            },
            timeout=180,
        )
        report["mac_native_wall_s"] = native_s
        report["mac_native_timings"] = native.get("timings")
        report["mac_native_tokens"] = native.get("tokens")
        report["mac_native_content"] = native.get("content")
        report["tokens_match_native"] = report["handoff_tokens"] == report["mac_native_tokens"]

        ht = report["handoff_timings"] or {}
        report["cache_n"] = ht.get("cache_n")
        report["prompt_n"] = ht.get("prompt_n")
        report["valid_cache"] = ht.get("cache_n") == 2111 and ht.get("prompt_n") == 1

        prod_t = report["producer_timings"] or {}
        spark_prefill_s = (prod_t.get("prompt_ms") or 0) / 1000.0
        save_ms = (saved or {}).get("timings", {}).get("save_ms")
        restore_ms = (restored or {}).get("timings", {}).get("restore_ms")
        handoff_prompt_s = (ht.get("prompt_ms") or 0) / 1000.0
        ttft_s = spark_prefill_s + (save_ms or 0) / 1000.0 + copy_s + (restore_ms or 0) / 1000.0 + handoff_prompt_s
        mac_cold_s = (report["mac_native_timings"] or {}).get("prompt_ms", 0) / 1000.0
        report["spark_prefill_s"] = spark_prefill_s
        report["ttft_split_s"] = ttft_s
        report["mac_cold_prefill_s"] = mac_cold_s
        report["ttft_speedup"] = (mac_cold_s / ttft_s) if ttft_s else None
        report["decode_tok_s"] = (ht.get("predicted_per_second") or ht.get("predicted_n") and None)

        out = ROOT / "report.json"
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({
            "valid_cache": report["valid_cache"],
            "tokens_match_native": report["tokens_match_native"],
            "state_bytes": size,
            "copy_s": round(copy_s, 3),
            "copy_MBps": round(report["copy_MBps"] or 0, 1),
            "spark_prefill_s": round(spark_prefill_s, 3),
            "mac_cold_prefill_s": round(mac_cold_s, 3),
            "ttft_split_s": round(ttft_s, 3),
            "ttft_speedup": round(report["ttft_speedup"] or 0, 2),
            "cache_n": report["cache_n"],
            "prompt_n": report["prompt_n"],
            "handoff_tokens": report["handoff_tokens"],
            "native_tokens": report["mac_native_tokens"],
            "spark_health_s": round(report["spark_health_s"], 1),
            "mac_health_s": round(report["mac_health_s"], 1),
        }, indent=2), flush=True)
    finally:
        if mac_proc and mac_proc.poll() is None:
            mac_proc.send_signal(signal.SIGTERM)
            try:
                mac_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                mac_proc.kill()
        if spark_pid:
            stop_spark(spark_pid)
        mac_log.close()


if __name__ == "__main__":
    main()
