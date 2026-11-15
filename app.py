"""Text Summarizer: Streamlit + Claude, with logging, rate limiting and a usage report."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import anthropic

# ---------- Configuration (everything comes from environment variables) ----------
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("CLAUDE_EFFORT", "low")  # low | medium | high: lower is faster and cheaper
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_HOUR", "5"))  # requests per user per hour
TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "60"))
MAX_INPUT_CHARS = 20_000
# Price in USD per 1 million tokens. Defaults match claude-opus-5-5; change them if you change the model.
PRICE_IN = float(os.environ.get("PRICE_INPUT_PER_M", "4.00"))
PRICE_OUT = float(os.environ.get("PRICE_OUTPUT_PER_M", "20.00"))

LOG_FILE = Path(__file__).parent / "logs" / "requests.jsonl"
WINDOW_SECONDS = 3600


# ---------- Structured logging: one JSON object per line ----------
def write_log(entry: dict) -> None:
    LOG_FILE.parent.mkdir(exist_ok=True)
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def read_logs() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    return [json.loads(line) for line in LOG_FILE.read_text().splitlines() if line.strip()]


# ---------- Rate limiter: in-memory {user: [timestamps]} shared by all sessions ----------
@st.cache_resource
def request_history() -> dict[str, list[float]]:
    return {}


def check_rate_limit(user: str) -> tuple[bool, int]:
    """Return (allowed, minutes_until_next_slot)."""
    history = request_history()
    now = time.time()
    recent = [t for t in history.get(user, []) if now - t < WINDOW_SECONDS]
    history[user] = recent
    if len(recent) >= RATE_LIMIT:
        wait = int((WINDOW_SECONDS - (now - recent[0])) / 60) + 1
        return False, wait
    recent.append(now)
    return True, 0


# ---------- The LLM call ----------
def summarize(text: str, style: str, failure_mode: str):
    api_key, timeout, retries = API_KEY, TIMEOUT_SECONDS, 1
    if failure_mode == "Revoked API key":
        api_key = "sk-ant-invalid-key-for-demo"
    elif failure_mode == "Timeout":
        timeout, retries = 0.001, 0  # 1 ms: guaranteed to time out
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=retries)
    return client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        output_config={"effort": EFFORT},
        # If the model declines a request, the API retries it on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system="You summarize text. Reply with only the summary, in the requested style.",
        messages=[{"role": "user", "content": f"Summarize the following text as {style}.\n\n---\n{text}"}],
    )


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, anthropic.APITimeoutError):
        return "The summarizer took too long to respond. Please try again in a moment."
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return "The summarizer is temporarily unavailable (configuration problem on our side). Please try again later."
    if isinstance(exc, anthropic.RateLimitError):
        return "The summarizer is very busy right now. Please try again in a few minutes."
    if isinstance(exc, anthropic.APIConnectionError):
        return "The summarizer could not be reached. Please try again in a moment."
    if isinstance(exc, RefusedError):
        return "The summarizer could not process this text. Please try different text."
    return "Something went wrong while summarizing. Please try again later."


class RefusedError(Exception):
    pass


def handle_request(user: str, text: str, style: str, failure_mode: str) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": user,
        "model": MODEL,
        "input_chars": len(text),
        "latency_ms": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "est_cost_usd": 0.0,
        "outcome": "ok",
        "error": None,
    }

    allowed, wait = check_rate_limit(user)
    if not allowed:
        entry["outcome"] = "rate_limited"
        write_log(entry)
        st.warning(
            f"Rate limit reached: {RATE_LIMIT} summaries per hour per user. "
            f"Try again in about {wait} minute(s)."
        )
        return

    start = time.perf_counter()
    try:
        with st.spinner("Summarizing..."):
            response = summarize(text, style, failure_mode)
        # Tokens are billed even if the model declines, so record them first.
        entry["input_tokens"] = response.usage.input_tokens
        entry["output_tokens"] = response.usage.output_tokens  # includes thinking tokens
        entry["est_cost_usd"] = round(
            entry["input_tokens"] / 1e6 * PRICE_IN + entry["output_tokens"] / 1e6 * PRICE_OUT, 6
        )
        if response.stop_reason == "refusal":
            raise RefusedError("model declined the request")
        summary = "".join(block.text for block in response.content if block.type == "text")
        if not summary:
            raise ValueError("empty response from model")
        st.subheader("Summary")
        st.write(summary)
    except Exception as exc:  # never show a stack trace to the user
        entry["outcome"] = "error"
        entry["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        st.error(friendly_error(exc))
    finally:
        entry["latency_ms"] = round((time.perf_counter() - start) * 1000)
        write_log(entry)


# ---------- Usage report ----------
def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * len(ordered)) - 1))
    return ordered[index]


def usage_page() -> None:
    st.title("Usage summary")
    logs = read_logs()
    if not logs:
        st.info("No requests logged yet.")
        return

    # Latency percentiles only count requests that actually reached the model.
    latencies = [e["latency_ms"] for e in logs if e["outcome"] != "rate_limited"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Requests", len(logs))
    col2.metric("p50 latency", f"{percentile(latencies, 50):,.0f} ms" if latencies else "n/a")
    col3.metric("p95 latency", f"{percentile(latencies, 95):,.0f} ms" if latencies else "n/a")
    col4.metric("Est. total cost", f"${sum(e['est_cost_usd'] for e in logs):.4f}")

    st.caption(
        f"Period: {logs[0]['timestamp'][:19]} to {logs[-1]['timestamp'][:19]} UTC. "
        f"Tokens: {sum(e['input_tokens'] for e in logs):,} in, "
        f"{sum(e['output_tokens'] for e in logs):,} out. "
        f"Cost estimated at ${PRICE_IN}/M input and ${PRICE_OUT}/M output tokens."
    )

    outcomes: dict[str, int] = {}
    for e in logs:
        outcomes[e["outcome"]] = outcomes.get(e["outcome"], 0) + 1
    st.subheader("Requests by outcome")
    st.bar_chart(outcomes, horizontal=True)

    st.subheader("Log entries (newest first)")
    st.dataframe(list(reversed(logs)), width="stretch")
    st.download_button("Download raw log (JSONL)", LOG_FILE.read_bytes(), "requests.jsonl")


# ---------- Summarizer page ----------
def summarizer_page() -> None:
    st.title("Text Summarizer")
    st.caption(f"Paste any text and get a short summary. Limit: {RATE_LIMIT} summaries per hour.")

    user = st.text_input("Your name", max_chars=40, placeholder="e.g. alex")
    style = st.selectbox("Summary style", ["3 bullet points", "one sentence", "a short paragraph"])
    text = st.text_area("Text to summarize", height=250, max_chars=MAX_INPUT_CHARS)

    with st.sidebar.expander("Failure simulation (demo)"):
        failure_mode = st.radio("Simulate", ["None", "Revoked API key", "Timeout"])

    if st.button("Summarize", type="primary"):
        if not user.strip():
            st.warning("Please enter your name first.")
        elif not text.strip():
            st.warning("Please paste some text to summarize.")
        else:
            handle_request(user.strip().lower(), text, style, failure_mode)


page = st.sidebar.radio("Page", ["Summarizer", "Usage summary"])
if page == "Summarizer":
    summarizer_page()
else:
    usage_page()
