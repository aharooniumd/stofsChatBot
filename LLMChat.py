import json
from llama_cpp import Llama
from datetime import date, datetime
from query import summarize_metric_by_date_lead, get_available_stations

# =========================
# Defaults for runtime
# =========================
DEFAULT_DATA_DIR = "s3://noaa-gestofs-pds/_post_processing/_metrics"
DEFAULT_PATH_TEMPLATE = "{data_dir}/{date}/**/*.{station}.cwl.csv"
DEFAULT_DATE_FMT = "%Y%m%d"
DEFAULT_LEAD_MIN = 0
DEFAULT_LEAD_MAX = 150


def get_all_available_stations(start_date, end_date, data_dir=DEFAULT_DATA_DIR, path_template=DEFAULT_PATH_TEMPLATE,
                               date_fmt=DEFAULT_DATE_FMT):
    return get_available_stations(
        start_date=start_date,
        end_date=end_date,
        data_dir=data_dir,
        path_template=path_template,
        date_fmt=date_fmt,
    )


# =========================
# System prompts
# =========================
system_prompt = """
You are a hybrid assistant with two modes.

MODE A — Function Call (metrics queries):
- Trigger ONLY when the user asks for metrics/analytics that require data retrieval or summarization (e.g., mentions like RMSD, Skil, Bias, RVal/Corr/MAE/MSE; stations; dates; lead times; “summarize”, “max/min”, “stats”, etc.).
- When triggered, reply ONLY with JSON in EXACTLY this format (no extra text):
{
  "function": "summarize_metric_by_date_lead",
  "args": {
    "start_date": "<YYYY-MM-DD>",
    "end_date": "<YYYY-MM-DD>",
    "lead_min": <number>,
    "lead_max": <number>,
    "metric": "<metric>",
    "stations": []
  }
}
- Defaults if omitted by the user:
  • start_date = today (YYYY-MM-DD)
  • end_date   = today (YYYY-MM-DD)
  • lead_min   = 0
  • lead_max   = 150
  • stations   = []
- "metric" must mirror the user’s requested metric string (e.g., "RMSD", "Skil", "Bias", "RVal").

MODE B — Normal Chat (everything else):
- Provide a brief, helpful natural-language reply. Do NOT output JSON.
- For small talk (e.g., “how are you?”), be polite and mention readiness to help with metrics analytics anytime.

Examples:
User: "What's the RMSD for station 8638901 on 2024-08-15?"
Assistant: {
  "function": "summarize_metric_by_date_lead",
  "args": { "start_date": "2024-08-15", "end_date": "2024-08-15", "lead_min": 0, "lead_max": 150, "metric": "RMSD", "stations": ["8638901"] }
}

User: "How are you doing?"
Assistant: "Thank you for asking—I'm ready to help you with retrieving metrics analytics at any time."

User: "Show me Skil last week for 8638610 and 8654467."
Assistant: {
  "function": "summarize_metric_by_date_lead",
  "args": { "start_date": "<today_minus_7>", "end_date": "<today>", "lead_min": 0, "lead_max": 150, "metric": "Skil", "stations": ["8638610","8654467"] }
}

User: "Explain what RMSD means."
Assistant: "Root Mean Square Deviation (RMSD) measures the average magnitude of errors between predictions and observations..."
"""

# General chat system for non-function replies
chat_system_prompt = "You are a concise, helpful assistant. Answer directly and briefly."


# =========================
# Helpers
# =========================
def parse_args_with_defaults(args):
    today = date.today().isoformat()
    args.setdefault("start_date", today)
    args.setdefault("end_date", today)
    args.setdefault("lead_min", DEFAULT_LEAD_MIN)
    args.setdefault("lead_max", DEFAULT_LEAD_MAX)
    stations = args.get("stations")
    if stations is None or len(stations) == 0:
        args["stations"] = []
    return args


def _extract_first_json_object(text: str) -> str:
    if not text:
        raise ValueError("Empty LLM output")

    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            inner_chunks = [parts[i] for i in range(1, len(parts) - 1)]
            candidate = max(inner_chunks, key=len)
            text = candidate.strip()
            first_newline = text.find("\n")
            if first_newline != -1 and first_newline < 16:
                maybe_lang = text[:first_newline].strip().lower()
                if maybe_lang in ("json", "javascript", "txt"):
                    text = text[first_newline + 1:].strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found in LLM output")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    return text[start:end].strip()
    raise ValueError("Unbalanced braces: could not find end of JSON object in LLM output")


def _to_jsonable(obj):
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None

    if pd is not None:
        if isinstance(obj, pd.DataFrame):
            import json as _json
            return _json.loads(obj.to_json(orient="records", date_format="iso"))
        if isinstance(obj, pd.Series):
            return _to_jsonable(obj.to_dict())
        if isinstance(obj, pd.Timestamp):
            return obj.to_pydatetime().isoformat()

    if np is not None:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return [_to_jsonable(x) for x in obj.tolist()]

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    return obj


def _strip_trailing_commas(json_text: str) -> str:
    import re
    return re.sub(r",\s*([}\]])", r"\1", json_text)


def call_llm_for_function(user_prompt):
    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
    output = llm(
        prompt=prompt,
        stop=["<|user|>", "<|system|>"],
        max_tokens=256,
        temperature=0.2,
    )["choices"][0]["text"].strip()

    json_text = _extract_first_json_object(output)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        fixed = _strip_trailing_commas(json_text)
        return json.loads(fixed)


def generate_human_answer(user_prompt, stats, args):
    answer_prompt = f"""
You are an assistant that summarizes results for metrics queries.
- Be concise and relevant to the user's question.
- If the user only asks for max, don't mention min, and vice versa.

User question:
{user_prompt}

Statistics dictionary:
{json.dumps(_to_jsonable(stats), ensure_ascii=False)}

Date range: {args.get('start_date')} to {args.get('end_date')}
Lead times: {args.get('lead_min')} to {args.get('lead_max')}
Metric: {args.get('metric')}
Stations: {args.get('stations')}
"""
    output = llm(
        prompt=f"<|system|>\n{answer_prompt}\n<|assistant|>\n",
        stop=["<|user|>", "<|system|>"],
        max_tokens=192,
        temperature=0.2,
    )["choices"][0]["text"].strip()
    return output


# =========================
# New: continuous chat loop with memory
# =========================
def chat_reply(history, user_msg):
    """
    Fallback normal chat (non-function) reply using conversation memory.
    `history` is a list of (role, content) tuples. Roles: "user" or "assistant".
    """
    # Build a simple chat-style prompt with memory
    parts = [f"<|system|>\n{chat_system_prompt}\n"]
    for role, content in history:
        if role == "user":
            parts.append(f"<|user|>\n{content}\n")
        else:
            parts.append(f"<|assistant|>\n{content}\n")
    parts.append(f"<|user|>\n{user_msg}\n<|assistant|>\n")
    prompt = "".join(parts)

    out = llm(
        prompt=prompt,
        stop=["<|user|>", "<|system|>"],
        max_tokens=256,
        temperature=0.3,
    )["choices"][0]["text"].strip()
    return out


def try_function_call(user_input):
    """
    Attempts the function-calling path.
    Returns (handled: bool, assistant_text: str)
      - handled=True when function call succeeded and produced outputs (keeps your original prints).
      - assistant_text is the human answer to append to chat history.
    Raises exceptions for genuine errors in your existing pipeline.
    """
    llm_response = call_llm_for_function(user_input)
    fn = llm_response["function"]
    args = parse_args_with_defaults(llm_response["args"])

    print(f"LLM JSON: function: {fn} with args: {args}")

    if fn == "summarize_metric_by_date_lead":
        # If stations not specified, populate with all available stations for the date(s)
        if not args.get("stations"):
            args["stations"] = get_all_available_stations(
                start_date=args["start_date"],
                end_date=args["end_date"]
            )
            # Keep this print to preserve your existing output
            print(f"Stations not specified, using all available stations: {args['stations']}")

        result = summarize_metric_by_date_lead(
            start_date=args["start_date"],
            end_date=args["end_date"],
            lead_min=args["lead_min"],
            lead_max=args["lead_max"],
            metric=args["metric"],
            stations=args["stations"],
            data_dir=DEFAULT_DATA_DIR,
            path_template=DEFAULT_PATH_TEMPLATE,
            date_fmt=DEFAULT_DATE_FMT,
            quiet_missing=False,
            tol=0.0,
        )
        print("Raw stats result:", result)
        print("\n---\n")
        answer = generate_human_answer(user_input, result, args)
        print(answer)
        return True, answer
    else:
        # Unknown function → treat as unhandled so we fall back to normal chat
        return False, ""


# =========================
# Instantiate llama.cpp
# =========================
# Instantiate llama.cpp
# n_ctx is for context window
# n_gpu_layers = -1 offloads all transformers on to gpu (full accel., you can change this number if you want to run parts on cpu)
llm = Llama(
    model_path="your/model/path",
    n_ctx=32768, n_gpu_layers=-1, n_threads=8, n_batch=4096
)


# =========================
# Continuous REPL
# =========================
def main():
    print("Interactive metrics/chat. Type 'quit' to exit.")
    history = []  # simple in-session memory: list of (role, content)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Bye.")
            break

        # First, try the function-calling path (keeps your original outputs exactly)
        try:
            handled, assistant_text = try_function_call(user_input)
        except Exception as e:
            # If the LLM tried to function-call but failed (e.g., invalid JSON),
            # or downstream raised, we fall back to normal chat with an error note.
            handled = False
            assistant_text = f"Error while processing function call: {e}"

        if handled:
            # Update chat memory with the user input and final human answer
            history.append(("user", user_input))
            history.append(("assistant", assistant_text))
            # Trim memory if it grows too large (simple cap)
            if len(history) > 40:
                history = history[-40:]
            continue

        # If not handled as a function call, do a normal chat turn using memory
        reply = chat_reply(history, user_input)
        print(reply)
        history.append(("user", user_input))
        history.append(("assistant", reply))
        if len(history) > 40:
            history = history[-40:]


if __name__ == "__main__":
    main()
