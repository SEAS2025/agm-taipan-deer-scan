"""
LLM assistant for AGM Deer Scan — Ollama, OpenAI, or local fallback.

Set OPENAI_API_KEY or run Ollama (ollama pull llama3.2) for full chat.
Without either, uses a context-aware local assistant.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional

SYSTEM_PROMPT = """You are the AI copilot for AGM Taipan Deer Scan — a thermal monocular
mounted on an ambulance for white-tailed deer detection near roads.

You help operators interpret live detections, tune sensitivity, explain thermal signatures,
and report YOLO training progress. Be concise and practical. Safety-first for roadside wildlife."""

FALLBACK_HINTS = [
    ("sensitivity", "Lower sensitivity = fewer false alerts. Start around 1.0 for open road, 0.7 at dusk."),
    ("thermal", "Deer appear as bright hot blobs against cooler background. Best at night or cool weather."),
    ("training", "Training uses visual + thermal deer images. Check the Training panel for live progress."),
    ("yolo", "After training completes, restart the scanner with --model agm_deer_ml/models/deer_thermal_best.pt"),
    ("alert", "Red DEER ALERT means 3 consecutive frames with deer-shaped hot signatures confirmed."),
]


class LLMAssistant:
    def __init__(
        self,
        ollama_host: str | None = None,
        ollama_model: str | None = None,
        openai_model: str | None = None,
    ):
        self.ollama_host = (ollama_host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_model = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        if self.openai_key:
            return "openai"
        try:
            req = urllib.request.Request(f"{self.ollama_host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return "ollama"
        except Exception:
            pass
        return "local"

    @property
    def backend(self) -> str:
        return self._backend

    def status(self) -> dict[str, Any]:
        return {
            "backend": self._backend,
            "ollama_host": self.ollama_host,
            "ollama_model": self.ollama_model,
            "openai_configured": bool(self.openai_key),
        }

    def chat(self, message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = context or {}
        system = SYSTEM_PROMPT + "\n\nLive context:\n" + json.dumps(context, indent=2)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ]
        try:
            if self._backend == "openai":
                reply = self._chat_openai(messages)
            elif self._backend == "ollama":
                reply = self._chat_ollama(messages)
            else:
                reply = self._chat_local(message, context)
            return {"ok": True, "reply": reply, "backend": self._backend}
        except Exception as e:
            fallback = self._chat_local(message, context)
            return {
                "ok": True,
                "reply": fallback + f"\n\n_(Text assistant unavailable: {e}. Using local helper.)_",
                "backend": "local",
            }

    def analyze_frame(
        self,
        jpeg_bytes: bytes,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = context or {}
        if self._backend == "openai":
            try:
                reply = self._vision_openai(jpeg_bytes, context)
                return {"ok": True, "reply": reply, "backend": "openai"}
            except Exception as e:
                return {"ok": True, "reply": self._analyze_local(context), "backend": "local", "note": str(e)}
        if self._backend == "ollama":
            try:
                reply = self._vision_ollama(jpeg_bytes, context)
                return {"ok": True, "reply": reply, "backend": "ollama"}
            except Exception:
                pass
        return {"ok": True, "reply": self._analyze_local(context), "backend": "local"}

    def _chat_openai(self, messages: list[dict]) -> str:
        body = json.dumps({"model": self.openai_model, "messages": messages, "max_tokens": 500}).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"].strip()

    def _chat_ollama(self, messages: list[dict]) -> str:
        body = json.dumps({"model": self.ollama_model, "messages": messages, "stream": False}).encode()
        req = urllib.request.Request(
            f"{self.ollama_host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        return data["message"]["content"].strip()

    def _vision_openai(self, jpeg_bytes: bytes, context: dict) -> str:
        b64 = base64.b64encode(jpeg_bytes).decode()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this thermal/live scan frame for white-tailed deer near a road. "
                            f"Detector context: {json.dumps(context)}"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            },
        ]
        body = json.dumps({"model": self.openai_model, "messages": messages, "max_tokens": 400}).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"].strip()

    def _vision_ollama(self, jpeg_bytes: bytes, context: dict) -> str:
        b64 = base64.b64encode(jpeg_bytes).decode()
        body = json.dumps(
            {
                "model": os.getenv("OLLAMA_VISION_MODEL", "llava"),
                "prompt": (
                    "You analyze thermal deer scans. Describe deer risk in this frame briefly.\n"
                    f"Context: {json.dumps(context)}\n"
                ),
                "images": [b64],
                "stream": False,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.ollama_host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        return data["response"].strip()

    def _chat_local(self, message: str, context: dict) -> str:
        msg = message.lower()
        parts = []

        if any(w in msg for w in ("status", "scan", "detect", "see")):
            parts.append(self._analyze_local(context))

        for key, hint in FALLBACK_HINTS:
            if key in msg:
                parts.append(hint)

        training = context.get("training") or {}
        if any(w in msg for w in ("train", "epoch", "dataset", "yolo", "model")):
            phase = training.get("phase", "idle")
            progress = training.get("progress", 0)
            log = training.get("last_log", "")
            parts.append(f"Training: **{phase}** ({progress}%). {log}")

        if not parts:
            parts.append(
                "I'm the local scan assistant (no cloud language model connected). "
                "Ask about sensitivity, thermal tips, or training status — or click **Analyze frame**. "
                "The deer detector is YOLO (computer vision), not this chat. "
                "For full text chat, run `ollama pull llama3.2` or set OPENAI_API_KEY."
            )
        return "\n\n".join(parts)

    def _analyze_local(self, context: dict) -> str:
        status = context.get("status", "unknown")
        deer = context.get("deer_hits", 0)
        fps = context.get("fps", 0)
        armed = context.get("armed", False)
        mode = "YOLO" if context.get("use_yolo") else "thermal heuristic"

        if armed:
            risk = "HIGH — confirmed deer alert active."
        elif deer > 0:
            risk = f"ELEVATED — tracking {deer} hot signature(s), not yet confirmed."
        else:
            risk = "LOW — no deer signatures in current frame."

        return (
            f"**Scan analysis** ({mode}, {fps:.1f} FPS)\n"
            f"Status: {status}\n"
            f"Risk: {risk}\n"
            f"White-tailed deer on roads appear as compact bright thermal blobs "
            f"with warmer core vs cooler edge — typical body temp ~38°C."
        )
