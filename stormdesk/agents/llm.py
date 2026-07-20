"""Minimal OpenAI-compatible chat client (urllib only, no extra deps)."""
from __future__ import annotations

import json
import re
import time
import urllib.request

from ..config import get_paths


class LLMClient:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 temperature: float = 0.3, max_tokens: int = 1600,
                 timeout: int = 240):
        p = get_paths()
        self.base_url = (base_url or p.llm_base_url).rstrip("/")
        self.model = model or p.llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.n_calls = 0
        self.n_prompt_tokens = 0
        self.n_completion_tokens = 0

    def chat(self, system: str, user: str, temperature: float | None = None,
             max_tokens: int | None = None, retries: int = 4) -> str:
        payload = dict(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        body = json.dumps(payload).encode()
        last = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    self.base_url + "/chat/completions", data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": "Bearer EMPTY"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read().decode())
                self.n_calls += 1
                u = resp.get("usage", {})
                self.n_prompt_tokens += u.get("prompt_tokens", 0)
                self.n_completion_tokens += u.get("completion_tokens", 0)
                return resp["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(min(2 ** attempt * 2.0, 30.0))
        raise RuntimeError(f"LLM call failed after {retries} tries: {last}")


def ask_json(llm: "LLMClient", system: str, user: str, retries: int = 2) -> dict:
    """Chat and parse a JSON reply; on parse failure, re-ask once."""
    last = None
    msg = user
    for _ in range(retries):
        reply = llm.chat(system, msg)
        try:
            return extract_json(reply)
        except (ValueError, json.JSONDecodeError) as e:
            last = e
            msg = (user + "\n\nYour previous reply could not be parsed as JSON "
                   f"({e}). Respond again with ONLY a single valid JSON object, "
                   "no comments, no placeholders.")
    raise ValueError(f"JSON parse failed after retries: {last}")


def extract_json(text: str) -> dict:
    """Extract the first balanced JSON object from an LLM reply."""
    text = re.sub(r"```(?:json)?", "", text)
    # strip // line comments (models sometimes copy them from the schema);
    # only when preceded by whitespace/comma/brace so "http://" survives
    text = re.sub(r"(?m)(?<=[\s,{}\[\]])//[^\n\"]*$", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON in reply: {text[:200]}")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
        elif ch == '"' and not esc:
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    frag = text[start:i + 1]
                    frag = re.sub(r",\s*([}\]])", r"\1", frag)  # trailing commas
                    return json.loads(frag)
    raise ValueError(f"unbalanced JSON in reply: {text[:200]}")
