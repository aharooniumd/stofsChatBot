import json
from llama_cpp import Llama
from datetime import date, datetime
from query import summarize_metric_by_date_lead, get_available_stations


# Defaults for runtime
DEFAULT_DATA_DIR = "metrics_data"
DEFAULT_PATH_TEMPLATE = "{data_dir}/{date}/*.{station}.cwl.csv"
DEFAULT_DATE_FMT = "%Y%m%d"
DEFAULT_LEAD_MIN = 0
DEFAULT_LEAD_MAX = 150

def get_all_available_stations(start_date, end_date, data_dir=DEFAULT_DATA_DIR, path_template=DEFAULT_PATH_TEMPLATE, date_fmt=DEFAULT_DATE_FMT):
    return get_available_stations(
        start_date=start_date,
        end_date=end_date,
        data_dir=data_dir,
        path_template=path_template,
        date_fmt=date_fmt,
    )

# Updated system prompt: only summarize_metric_by_date_lead, with defaults semantics
system_prompt = """
You are a function-calling assistant. Reply ONLY with JSON in this exact format:
{
    "function": "summarize_metric_by_date_lead",
    "args": {
        "start_date": "<YYYY-MM-DD>",
        "end_date": "<YYYY-MM-DD>",
        "lead_min": <number>,
        "lead_max": <number>,
        "metric": "<metric>",
        "stations": [<station_ids>]  // leave empty if the user did not specify any stations
    }
}

Rules:
- The function name MUST be "summarize_metric_by_date_lead".
- If the user does not specify dates, set both start_date and end_date to today's date (YYYY-MM-DD).
- If the user does not specify a lead range, use lead_min=0 and lead_max=150.
- If the user does not specify stations, set "stations" to an empty array [].
- "metric" must reflect the user's requested metric (e.g., "RMSD", "Skil", "Bias", "RVal", etc.).
- Output JSON only—no explanations or extra text.
"""

# Normalize/complete args with defaults
def parse_args_with_defaults(args):
    today = date.today().isoformat()
    args.setdefault("start_date", today)
    args.setdefault("end_date", today)
    args.setdefault("lead_min", DEFAULT_LEAD_MIN)
    args.setdefault("lead_max", DEFAULT_LEAD_MAX)
    # Keep empty list to signal "not specified" so we can resolve all stations
    stations = args.get("stations")
    if stations is None or len(stations) == 0:
        args["stations"] = []
    return args

def _extract_first_json_object(text: str) -> str:
    """
    Extract the first complete JSON object from text using brace matching.
    Handles strings and escape characters to avoid premature termination.
    Also strips Markdown code fences if present.
    """
    if not text:
        raise ValueError("Empty LLM output")

    # If wrapped in a markdown code fence, try to unwrap it.
    # Supports ```json ... ``` or ``` ... ```
    if "```" in text:
        parts = text.split("```")
        # look for the largest content chunk between fences
        if len(parts) >= 3:
            # common pattern: prefix, (lang), content, suffix
            # pick the content part; handle cases where language tag is present
            # Prefer the longest inner part to reduce risk
            inner_chunks = [parts[i] for i in range(1, len(parts) - 1)]
            candidate = max(inner_chunks, key=len)
            text = candidate.strip()
            # If language tag present, drop the first line
            first_newline = text.find("\n")
            if first_newline != -1 and first_newline < 16:  # likely a lang tag like 'json'
                maybe_lang = text[:first_newline].strip().lower()
                if maybe_lang in ("json", "javascript", "txt"):
                    text = text[first_newline + 1 :].strip()

    # Find the first object start
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
            # continue scanning inside string
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
            # other characters outside strings: ignore

    raise ValueError("Unbalanced braces: could not find end of JSON object in LLM output")

def call_llm(user_prompt):
    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
    output = llm(
        prompt=prompt,
        stop=["<|user|>", "<|system|>"],
        max_tokens=256,
        temperature=0.2,
    )["choices"][0]["text"].strip()
    try:
        json_text = _extract_first_json_object(output)
        return json.loads(json_text)
    except Exception:
        print(f"Could not parse LLM output: {output}")
        raise

def _to_jsonable(obj):
    """
    Convert pandas/numpy/datetime-rich structures into JSON-serializable
    Python primitives. DataFrame -> list[dict] with ISO8601 timestamps.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None

    # pandas-specific types
    if pd is not None:
        if isinstance(obj, pd.DataFrame):
            # Use pandas' own JSON handling for timestamps
            import json as _json
            return _json.loads(obj.to_json(orient="records", date_format="iso"))
        if isinstance(obj, pd.Series):
            return _to_jsonable(obj.to_dict())
        if isinstance(obj, pd.Timestamp):
            return obj.to_pydatetime().isoformat()

    # numpy types
    if np is not None:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return [_to_jsonable(x) for x in obj.tolist()]

    # datetime/date
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # containers
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    # already serializable (or unknown but hopefully OK)
    return obj

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

def main():
    user_input = input("Ask your metrics question: ")
    llm_response = call_llm(user_input)
    fn = llm_response["function"]
    args = parse_args_with_defaults(llm_response["args"])
    print(f"LLM JSON: function: {fn} with args: {args}")

    if fn == "summarize_metric_by_date_lead":
        # If stations not specified, populate with all available stations for the date(s)
        if len(args.get("stations")) == 0 or args.get("stations") is None:
            args["stations"] = get_all_available_stations(
                start_date=args["start_date"],
                end_date=args["end_date"]
            )

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
    else:
        print("Unknown function.")

# Instantiate llama.cpp
# n_ctx is for context window
# n_gpu_layers = -1 (on mac runs the model on metal gpu)
llm = Llama(
    model_path="your/model/path",
    n_ctx=32768, n_gpu_layers=-1, n_threads=8, n_batch=4096
)
if __name__ == "__main__":
    main()
