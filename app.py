import os
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")

if not API_KEY:
    raise ValueError(
        "ERROR: MISTRAL_API_KEY environment variable not found.\n"
        "Go to HuggingFace Space → Settings → Secrets → Add Secret:\n"
        "Key = MISTRAL_API_KEY\nValue = your_api_key"
    )

model = init_chat_model(
    "mistral-large-latest",
    model_provider="mistralai",
    api_key=API_KEY
)

system_template = """
You are an intelligent QA assistant for SIMULIA PowerFLOW / PowerACOUSTICS.

You help users with:
- Understanding simulation logs (errors, warnings, convergence issues).
- Suggesting likely root causes and next steps.
- Recommending QA test ideas and debugging workflows.
- Explaining commands and options in simple, clear language.

If log analytics are provided (most common errors, commands, patterns), use them as additional context.
If something is unclear or missing, ask a brief clarification question.
If you don't know, say so honestly instead of inventing details.
Keep answers concise, practical, and engineer-friendly.
"""

user_template = """
{maybe_context}

User question:
{user_question}
"""

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", user_template)]
)

ERROR_KEYWORDS = ["ERROR", "FATAL", "EXCEPTION", "ABORT", "FAIL"]
COMMAND_MIN_LENGTH = 3

def analyze_logs(texts: List[str]) -> Tuple[Counter, Counter, Dict[str, List[str]], Dict[str, Counter]]:
    all_lines: List[str] = []
    for t in texts:
        all_lines.extend(t.splitlines())

    cmd_counter = Counter()
    error_counter = Counter()
    error_examples: Dict[str, List[str]] = defaultdict(list)
    error_command_cooccurrence: Dict[str, Counter] = defaultdict(Counter)

    previous_command = None

    for line in all_lines:
        raw = line.rstrip("\n")
        stripped = raw.strip()
        if not stripped:
            continue

        tokens = stripped.split()
        if tokens:
            first_token = tokens[0]
            if len(first_token) >= COMMAND_MIN_LENGTH and not first_token.startswith("#"):
                cmd_counter[first_token] += 1
                previous_command = first_token

        if any(k in raw for k in ERROR_KEYWORDS):
            key = re.sub(r"\d+", "<NUM>", raw)
            key = re.sub(r"\s+", " ", key).strip()
            error_counter[key] += 1

            if len(error_examples[key]) < 3:
                error_examples[key].append(raw)

            if previous_command:
                error_command_cooccurrence[key][previous_command] += 1

    return cmd_counter, error_counter, error_examples, error_command_cooccurrence

def build_context_from_analytics(
    cmd_counter: Counter,
    error_counter: Counter,
    error_examples: Dict[str, List[str]],
    error_command_cooccurrence: Dict[str, Counter],
    top_n: int = 5,
) -> str:
    parts = []

    if cmd_counter:
        parts.append("Most used commands in the logs:")
        for cmd, cnt in cmd_counter.most_common(top_n):
            parts.append(f"- {cmd}: {cnt} occurrences")
        parts.append("")

    if error_counter:
        parts.append("Most common error patterns:")
        for err, cnt in error_counter.most_common(top_n):
            parts.append(f"- {cnt}×: {err}")
        parts.append("")

    if error_command_cooccurrence:
        parts.append("Patterns: commands frequently preceding errors:")
        for err, cmd_counts in list(error_command_cooccurrence.items())[:top_n]:
            top_cmds = ", ".join(
                f"{c}× {cmd}" for cmd, c in cmd_counts.most_common(3)
            )
            parts.append(f"- Error: {err}")
            parts.append(f"  Likely associated commands: {top_cmds}")
        parts.append("")

    if error_examples:
        parts.append("Representative error lines (examples):")
        count = 0
        for err, examples in error_examples.items():
            if count >= top_n:
                break
            parts.append(f"- Pattern: {err}")
            for ex in examples:
                parts.append(f"    example: {ex}")
            count += 1

    return "\n".join(parts) if parts else ""

def call_llm(user_question: str, analytics_context: str = "") -> str:
    if analytics_context:
        maybe_context = (
            "Here is analytics computed from the uploaded PowerFLOW logs.\n"
            "Use it if it helps answer the question:\n\n" + analytics_context
        )
    else:
        maybe_context = "No log context was provided."

    prompt = prompt_template.invoke(
        {"user_question": user_question, "maybe_context": maybe_context}
    )
    response = model.invoke(prompt)
    return response.content

st.set_page_config(page_title="PowerFLOW QA Assistant", page_icon=None, layout="wide")
st.title("SIMULIA PowerFLOW QA Assistant")
st.caption("LLM-powered assistant with log analytics for simulation debugging.")

st.markdown("---")

st.sidebar.header("Log Upload & Analytics")
uploaded_files = st.sidebar.file_uploader(
    "Upload PowerFLOW / PowerACOUSTICS logs",
    type=["log", "txt", "out"],
    accept_multiple_files=True,
)

log_texts: List[str] = []
analytics_context = ""

if uploaded_files:
    for f in uploaded_files:
        content = f.read().decode("utf-8", errors="ignore")
        log_texts.append(content)

    st.sidebar.success(f"{len(uploaded_files)} files loaded.")

    cmd_stats, err_stats, err_examples, err_cmd_patterns = analyze_logs(log_texts)
    analytics_context = build_context_from_analytics(
        cmd_stats, err_stats, err_examples, err_cmd_patterns
    )
else:
    st.sidebar.info("No logs uploaded yet. You can still ask questions.")

tab_chat, tab_analytics = st.tabs(["QA Assistant", "Log Analytics"])

with tab_chat:
    st.subheader("Ask about your simulation or logs")

    user_input = st.text_area("Your question or log snippet:", height=150)
    ask = st.button("Ask Assistant")

    if ask and user_input.strip():
        with st.spinner("Analyzing and responding..."):
            answer = call_llm(user_input, analytics_context=analytics_context)
        st.markdown("### Response")
        st.write(answer)

with tab_analytics:
    st.subheader("Log Analytics")

    if uploaded_files:
        if cmd_stats:
            st.markdown("### Most Used Commands")
            top_cmds = cmd_stats.most_common(15)
            st.table({"Command": [c for c, _ in top_cmds], "Count": [n for _, n in top_cmds]})

        if err_stats:
            st.markdown("### Most Common Error Patterns")
            top_errs = err_stats.most_common(15)
            st.table({"Error Pattern": [e for e, _ in top_errs], "Count": [n for _, n in top_errs]})

        if err_cmd_patterns:
            st.markdown("### Command–Error Associations")
            rows = []
            for err, cmd_counts in err_cmd_patterns.items():
                top_cmds = ", ".join(f"{cmd} ({cnt}×)" for cmd, cnt in cmd_counts.most_common(3))
                rows.append({"Error Pattern": err, "Likely Commands": top_cmds})
            st.table(rows[:20])
    else:
        st.info("Upload logs to view analytics.")
