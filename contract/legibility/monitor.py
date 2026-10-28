"""Client for the frozen monitor: a vLLM server (the base model, thinking off) reached over HTTP.

    monitor = Monitor()                               # URL from $MONITOR_URL, else from --monitor-url-file, else $LLM_BASE_URL
    answers, raw = monitor.judge(chains, names)       # one {behavior: probability | None} per chain
    answers, raw = monitor.judge(chains, names, outputs)   # diagnostic: the monitor also sees each final answer

Training jobs run the monitor on a second GPU of the same allocation (scripts/train_legibility.sbatch starts it and
exports MONITOR_URL); evaluation jobs point it at the vLLM server that scripts/serve_and_run.sbatch started.
"""
import os
import re

from contract.legibility.prompts import MONITOR_INTRO, MONITOR_INTRO_WITH_OUTPUT, MONITOR_NUMBERS, MONITOR_SYSTEM_PROMPT, MONITOR_YESNO
from contract.llm import LLM
from contract.run_tasks import parse_precommit


def run_tests_block(text):
    """The `def run_tests():` function in `text` (code or a message containing code), up to the next unindented line."""
    m = re.search(r"def run_tests\(\):.*?(?=\n(?=\S)|\Z)", text or "", re.DOTALL)
    return m.group(0).strip() if m else None


class Monitor:
    def __init__(self, model="Qwen/Qwen3-4B", url=None, url_file=None, elicitation="yesno", thinking=False, think_budget=1024, max_tokens=64, workers=64,
                 statements=None, context=None):
        """`elicitation`: "yesno" asks one statement at a time and reads P(yes) off the first answer token (thinking
        off); "numbers" asks for the whole list of probabilities in one reply, optionally after `think_budget` tokens
        of the monitor's own reasoning (`thinking`).  `statements` / `context`: the environment's rubric and its
        description of how credit is awarded (Env.monitor_statements / Env.monitor_context; LeetCode's by default)."""
        from contract.envs.leetcode import LeetCodeEnv
        self.statements = statements or LeetCodeEnv.monitor_statements
        self.context = context or LeetCodeEnv.monitor_context
        if url is None and url_file and os.path.exists(url_file):
            url = open(url_file).read().strip()
        url = url or os.environ.get("MONITOR_URLS") or os.environ.get("MONITOR_URL") or os.environ.get("LLM_BASE_URL")
        self.elicitation, self.names = elicitation, None
        self.llm = LLM(model=model, base_url=url, thinking=thinking, think_budget=think_budget if thinking else None,
                       temperature=0.0 if not thinking else 0.7, max_tokens=max_tokens, workers=workers)

    def intro(self, chain, output, tests=None):
        chain = (chain or "(empty)").strip()
        if output is None:
            text = MONITOR_INTRO.format(chain=chain, context=self.context)
        else:
            text = MONITOR_INTRO_WITH_OUTPUT.format(chain=chain, output=(output or "(empty)").strip(), context=self.context)
        if tests is not None:  # the environment's evidence block (Env.monitor_evidence), e.g. run_tests() before / after
            text += tests
        return text

    def messages(self, chain, names, output=None, statement=None, tests=None):
        """One query: all of `names` at once (numbers), or the single `statement` (yes/no)."""
        if statement is None:
            tail = MONITOR_NUMBERS.format(statements="\n".join(f"{i + 1}. {self.statements[b]}" for i, b in enumerate(names)))
        else:
            tail = MONITOR_YESNO.format(statement=self.statements[statement])
        return [{"role": "system", "content": MONITOR_SYSTEM_PROMPT}, {"role": "user", "content": self.intro(chain, output, tests) + tail}]

    def judge(self, chains, names, outputs=None, tests=None):
        """Probabilities per statement for each chain (None where the monitor's answer did not parse), plus raw texts.
        With `outputs` (one final answer per chain) the monitor sees the answer too; with `tests` (one evidence block per
        chain, from Env.monitor_evidence) it sees e.g. the grading function before and after."""
        outputs = outputs if outputs is not None else [None] * len(chains)
        tests = tests if tests is not None else [None] * len(chains)
        if self.elicitation == "numbers":
            outs = self.llm.chat_many([self.messages(c, names, o, tests=t) for c, o, t in zip(chains, outputs, tests)], n=1)
            raw = [o[0]["content"] for o in outs]
            return [dict(zip(names, parse_precommit(r, "prob", len(names)))) for r in raw], raw
        probs = self.llm.next_token_probs_many([self.messages(c, names, o, statement=b, tests=t) for c, o, t in zip(chains, outputs, tests) for b in names])
        answers, raw = [], []
        for i in range(len(chains)):
            a, r = {}, {}
            for b, dist in zip(names, probs[i * len(names):(i + 1) * len(names)]):
                yes = sum(v for t, v in dist.items() if t.strip().lower().startswith("yes"))
                no = sum(v for t, v in dist.items() if t.strip().lower().startswith("no"))
                a[b] = yes / (yes + no) if yes + no > 0 else None
                r[b] = {"yes": round(yes, 4), "no": round(no, 4)}
            answers.append(a); raw.append(str(r))
        return answers, raw
