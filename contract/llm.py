"""Thin client for independent vLLM replicas speaking the OpenAI chat API.

All model access in this project goes through this class so that the same
code can talk to a local vLLM server on the cluster (the default) or to any
other OpenAI-compatible endpoint. LLM_BASE_URLS supplies comma-separated replica URLs;
LLM_WORKERS controls concurrency. Single-server LLM_BASE_URL remains supported.
"""
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle

from openai import OpenAI
from transformers import AutoTokenizer  # imported here, not lazily: a first import from many threads at once fails

from contract.prompts import THINK_BUDGET_STOP


class LLM:
    def __init__(
        self,
        model="Qwen/Qwen3-4B",
        base_url=None,
        thinking=False,
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
        workers=None,
        think_budget=None,
        tokenizer=None,
    ):
        base_url = base_url or os.environ.get("LLM_BASE_URLS") or os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
        self.clients = cycle([OpenAI(base_url=url.strip(), api_key="none", timeout=1800, max_retries=5)
                              for url in base_url.split(",")])
        self.model = model
        self.thinking = thinking
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.workers = workers if workers is not None else int(os.environ.get("LLM_WORKERS", 64))
        self.think_budget = think_budget  # thinking mode only: max reasoning tokens before the block is force-closed
        # Renders the chat template for the continuation.  `tokenizer` names the base model when `model` is a served
        # adapter alias (vLLM --lora-modules rl=...), which is not a Hugging Face id.
        self.tok = AutoTokenizer.from_pretrained(tokenizer or model) if think_budget else None

    def chat(self, messages, n=1, thinking=None, think_budget=None, max_tokens=None, stop_text=THINK_BUDGET_STOP):
        """Sample n completions. Returns a list of {content, reasoning, finish_reason}.
        `thinking`, `think_budget` and `max_tokens` override the instance defaults for this call (a commitment turn
        generated without a chain, or with a shorter one and a short answer)."""
        thinking = self.thinking if thinking is None else thinking
        think_budget = self.think_budget if think_budget is None else think_budget
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        client = next(self.clients)
        if thinking and think_budget:
            return [self.chat_budgeted(messages, think_budget, max_tokens, stop_text, client=client) for _ in range(n)]
        r = client.chat.completions.create(
            model=self.model,
            messages=messages,
            n=n,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_tokens,
            # Qwen3-specific: the chat template takes an enable_thinking flag.
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        return [
            {
                "content": c.message.content or "",
                # vLLM has used both field names for the parsed <think> block across versions.
                "reasoning": getattr(c.message, "reasoning_content", None) or getattr(c.message, "reasoning", None),
                "finish_reason": c.finish_reason,
            }
            for c in r.choices
        ]

    def chat_budgeted(self, messages, think_budget, max_tokens, stop_text, client=None):
        """Thinking with a token budget, as in Qwen3's thinking_budget recipe: reason for at most `think_budget`
        tokens; if the block is still open, append THINK_BUDGET_STOP, close it, and let the model answer with up to
        `max_tokens` more.  The continuation goes through the raw completions endpoint with the chat template rendered
        by the model's tokenizer, because Qwen3's template rewrites <think> blocks in assistant messages (so the chat
        endpoint's `continue_final_message` fails).  The <think> markers and the stop string are the one Qwen-specific
        part; another model family would need its own here."""
        client = client or next(self.clients)  # keep reasoning and answer continuation on the same replica
        r = client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, top_p=self.top_p, max_tokens=think_budget,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}})
        c = r.choices[0]
        reasoning = getattr(c.message, "reasoning_content", None) or getattr(c.message, "reasoning", None) or ""
        if c.finish_reason != "length":
            return {"content": c.message.content or "", "reasoning": reasoning, "finish_reason": c.finish_reason, "think_forced": False}
        forced = not c.message.content  # block still open at the budget (a non-empty content means the answer got cut instead)
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        prefix = f"<think>\n{reasoning}\n{stop_text}\n</think>\n\n" if forced else f"<think>\n{reasoning}\n</think>\n\n{c.message.content}"
        r2 = client.completions.create(model=self.model, prompt=prompt + prefix, temperature=self.temperature, top_p=self.top_p,
                                            max_tokens=max_tokens, stop=[self.tok.eos_token])
        c2 = r2.choices[0]
        content = c2.text if forced else (c.message.content or "") + c2.text
        return {"content": content, "reasoning": reasoning + ("\n" + stop_text if forced else ""), "finish_reason": c2.finish_reason, "think_forced": forced}

    def iter_chat(self, message_lists, n=1, **kw):
        """Yield in input order while workers keep generating ahead of the consumer."""
        with ThreadPoolExecutor(self.workers) as ex:
            yield from ex.map(lambda m: self.chat(m, n=n, **kw), message_lists)

    def chat_many(self, message_lists, n=1, cache_path=None, **kw):
        """Optionally retain frozen-teacher responses across interrupted warm-up sampling jobs."""
        if cache_path is None:
            return list(self.iter_chat(message_lists, n=n, **kw))
        # Include the index so repeated prompts still receive independent samples.
        config = [self.model, self.thinking, self.think_budget, self.max_tokens, self.temperature, self.top_p]
        keys = [hashlib.sha256(json.dumps([config, i, m, n, kw], sort_keys=True).encode()).hexdigest()
                for i, m in enumerate(message_lists)]
        saved = {}
        if os.path.exists(cache_path):
            with open(cache_path, "rb+") as f:
                for line in f:
                    if not line.endswith(b"\n"):
                        f.truncate(f.tell() - len(line))
                        break
                    row = json.loads(line)
                    saved[row["key"]] = row["response"]
        pending = [i for i, key in enumerate(keys) if key not in saved]
        print(f"teacher responses: {len(keys) - len(pending)} saved, {len(pending)} remaining", flush=True)
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "a") as f:
            for i, response in zip(pending, self.iter_chat([message_lists[i] for i in pending], n=n, **kw)):
                saved[keys[i]] = response
                f.write(json.dumps({"key": keys[i], "response": response}) + "\n")
                f.flush()
        return [saved[key] for key in keys]

    def next_token_probs(self, messages, top=20):
        """Probability of each candidate first token of the assistant's reply.

        Thinking is always disabled here so that the first generated token is
        the answer itself rather than the start of a reasoning block.
        """
        r = next(self.clients).chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=top,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        top_logprobs = r.choices[0].logprobs.content[0].top_logprobs
        return {t.token: math.exp(t.logprob) for t in top_logprobs}

    def next_token_probs_many(self, message_lists, top=20):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda m: self.next_token_probs(m, top=top), message_lists))
