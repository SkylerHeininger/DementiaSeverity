"""
Skyler Heininger
10/31/2026

Generates a transcript, focused on a couple of pieces.

One, the topic is around dementia care. We can maybe inject some specific topics for the LLM 
to generate through the conversation. Probably focus on max like 100 turns, where the turn length
should be pretty variable. Typically in real visits, the doctor may lead most of the conversation, with longer turns,
but sometimes the patient/caregiver may take the lead.

Some inputs to consider for this script are to input dementia severity, which would automatically throttle the
speaker roles, repetition, coherence, etc, that we have seen from our results. For testing purposes, would be good
to provide an example of normal_to_mild, moderate, and severe dementia.

Some concerns are the LLM to actually generate more patient turns especially at higher severity. At high severity, you
usually see very little patient involvement directly, both because it becomes more difficult to communicate and follow
the conversation with worsening dementia, but also that doctors communicate more with the caregiver at higher dementia levels.

Turns should also for the most part bounce from provider to caregiver or patient, and back to provider.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import pandas as pd
import requests


# When writing out descriptions I kept thinking about patients whose recordings I had heard. 
# I did not want to leak anything, so just had an LLM generate these descriptions.
PERSONAS = {
    "normal_to_mild": {
        "patient_id": "P1", "patient": "Margaret Ellis", "age": 74,
        "background": "a retired middle-school teacher who lives with her husband and still manages most of her own day",
        "caregiver": "Tom Ellis", "relationship": "husband", "provider": "Dr. Patel",
        "situation": "a routine follow-up memory clinic visit; she has noticed occasional lapses, such as misplacing keys, "
                     "forgetting a friend's name for a moment, and missing an appointment once",
    },
    "moderate": {
        "patient_id": "P2", "patient": "Robert Alvarez", "age": 79,
        "background": "a retired machinist who lives with his wife and used to handle the family finances",
        "caregiver": "Linda Alvarez", "relationship": "wife", "provider": "Dr. Patel",
        "situation": "a follow-up memory clinic visit; he repeats questions and stories, has trouble with bills and "
                     "medications, sometimes gets confused about the day, and his wife has taken over many tasks",
    },
    "severe": {
        "patient_id": "P3", "patient": "Dorothy Whitaker", "age": 85,
        "background": "a former church organist who lives with her daughter, who has become her full-time caregiver",
        "caregiver": "Karen Whitaker", "relationship": "daughter", "provider": "Dr. Patel",
        "situation": "a follow-up memory clinic visit; she needs help with bathing, dressing, and eating, speaks very little, "
                     "is sometimes restless at night, and her daughter is exhausted",
    },
}

PROFILES = {
    "normal_to_mild": {
        "assumed_dsrs": 9, "p_patient": 0.62,
        "patient_latency": (0.6, 1.8), "patient_wps": 2.4,
        "guide": "{patient} speaks in full, fluent sentences and follows the conversation easily. She notices occasional lapses "
                 "(misplaced items, a forgotten name, a missed appointment) and is a little worried or embarrassed about them. "
                 "She sometimes pauses to find a word but recovers. She answers questions directly and often adds detail. "
                 "{caregiver} adds small corrections or concerns but mostly lets her speak.",
    },
    "moderate": {
        "assumed_dsrs": 27, "p_patient": 0.42,
        "patient_latency": (1.2, 3.5), "patient_wps": 2.0,
        "guide": "{patient} answers some questions himself in short, sometimes vague sentences, but often loses the thread, repeats a "
                 "question or story he already told, mixes up dates and details, and needs prompting. He can get defensive or insist "
                 "things are fine. {caregiver} often steps in to answer, correct, or fill in details. The doctor speaks to both, "
                 "but turns to {caregiver} more and more for specifics.",
    },
    "severe": {
        "assumed_dsrs": 44, "p_patient": 0.13,
        "patient_latency": (2.5, 8.0), "patient_wps": 1.5,
        "guide": "{patient} rarely contributes. When she does, it is a word or a short fragment, sometimes unrelated to the question, "
                 "repeated, or a social phrase such as 'thank you' or 'oh yes'. She cannot give a history or follow multi-step "
                 "questions. {caregiver} answers nearly everything, describes day-to-day function (help with bathing, dressing, eating, "
                 "continence, restlessness at night, wandering) and talks about how tiring it is. The doctor speaks mostly to "
                 "{caregiver}, and now and then greets or reassures {patient} with simple, warm, short sentences.",
    },
}

FALLBACK_TEXT = {"provider": "Okay, let's keep going.", "patient": "Mm-hm.", "caregiver": "Yes, that's right."}


def plan_roles(n_turns, p_patient, rng):
    roles = ["provider" if i % 2 == 0 else "patient" if rng.random() < p_patient else "caregiver" for i in range(n_turns)]
    caregiver_slots = [i for i, r in enumerate(roles) if r == "caregiver"]
    rng.shuffle(caregiver_slots)
    while roles.count("patient") < 3 and caregiver_slots:
        roles[caregiver_slots.pop()] = "patient"
    return roles


def build_prompt(turns, roles, lines, persona, profile):
    names = {"patient": persona["patient"], "caregiver": persona["caregiver"], "provider": persona["provider"]}
    guide = profile["guide"].format(patient=persona["patient"], caregiver=persona["caregiver"])
    spec = []
    for i in turns:
        if i == 0:
            action = "greets everyone and asks how things have been since the last visit"
        elif i == len(roles) - 1:
            action = "wraps up the visit, mentions follow-up, and says goodbye"
        elif roles[i] == "provider":
            action = f"speaks to {names[roles[i + 1]]}"
        else:
            action = f"responds to {names['provider']}"
        spec.append(f"{i + 1}. [{names[roles[i]]}, {roles[i]}] {action}")
    recent = "\n".join(lines[-10:]) or "(the visit is just starting)"
    return (
        "Write part of a realistic spoken transcript of a dementia clinic follow-up visit. Everyone in it is fictional.\n\n"
        f"Patient: {persona['patient']}, age {persona['age']}, {persona['background']}.\n"
        f"Caregiver: {persona['caregiver']}, the patient's {persona['relationship']}.\n"
        f"Clinician: {persona['provider']}.\n"
        f"Situation: {persona['situation']}.\n\n"
        f"How the people speak: {guide}\n\n"
        "Style: natural spoken language with contractions, hesitations, and occasional incomplete sentences. Be as succinct as "
        "possible, like real conversation: most turns are only a few words, and many are a single word or phrase (for example "
        "'Yes.', 'No, not really.', 'Mm-hm.', 'Okay.'). The clinician's questions usually get brief answers. When the patient or the "
        "caregiver asks the clinician something, the clinician's answer is usually longer, a sentence or a few. Only occasionally does "
        "anyone speak for more than a couple of sentences. "
        "No stage directions, no actions in brackets or asterisks, no speaker names inside the text. "
        "Choose realistic dementia-care topics as the visit goes on.\n\n"
        f"Conversation so far:\n{recent}\n\n"
        "Write the next turns. Each line below gives the turn number, who speaks, and what they do. "
        "Reply with exactly one line per turn, in the form `<number>. <what the person says>`, and nothing else.\n\n"
        + "\n".join(spec)
    )


def chat(prompt, args):
    system = "You write realistic, natural-sounding spoken dialogue for synthetic transcripts used in software demos."
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    body = {"model": args.model, "temperature": args.temperature, "max_tokens": args.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
    response = requests.post(args.base_url.rstrip("/") + "/chat/completions", json=body, headers=headers, timeout=args.timeout)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"] or ""


def resolve_model(args):
    try:
        response = requests.get(args.base_url.rstrip("/") + "/models", timeout=30,
                                headers={"Authorization": f"Bearer {args.api_key}"} if args.api_key else {})
        ids = [m["id"] for m in response.json()["data"]]
    except Exception:
        return args.model
    if args.model in ids:
        return args.model
    matches = [i for i in ids if args.model.lower() in i.lower()]
    if not matches:
        raise SystemExit(f"model {args.model!r} not found. Available: {ids}")
    print(f"using model id {matches[0]!r} for {args.model!r}")
    return matches[0]


def clean_text(text, persona):
    text = re.sub(r"\*[^*]*\*|\[[^\]]*\]|\([^)]*\)", " ", text)
    for name in (persona["patient"], persona["caregiver"], persona["provider"]):
        text = re.sub(rf"^\s*{re.escape(name)}\s*[:\-]\s*", "", text)
    return re.sub(r"\s+", " ", text).strip().strip('"').strip()


def parse_lines(reply, wanted, persona):
    found = {}
    for line in reply.splitlines():
        match = re.match(r"^\s*`?(\d+)[\.\):]\s*(.+?)`?\s*$", line)
        if match and int(match.group(1)) - 1 in wanted:
            text = clean_text(match.group(2), persona)
            if text:
                found[int(match.group(1)) - 1] = text
    return found


def request_turns(turns, roles, lines, persona, profile, args):
    found = {}
    for _ in range(args.retries + 1):
        missing = [i for i in turns if i not in found]
        if not missing:
            break
        reply = chat(build_prompt(missing, roles, lines, persona, profile), args)
        found.update(parse_lines(reply, set(missing), persona))
    return found


def add_timing(roles, texts, profile, rng):
    clock, rows = 0.0, []
    for i, (role, text) in enumerate(zip(roles, texts)):
        if i == 0 or roles[i - 1] == role:
            clock += rng.uniform(0.1, 0.5)
        else:
            clock += rng.uniform(*profile["patient_latency"]) if role == "patient" else rng.uniform(0.3, 1.0)
        start = clock
        clock += len(text.split()) / (profile["patient_wps"] if role == "patient" else 2.6) + 0.3
        rows.append((round(start, 2), round(clock, 2)))
    return rows


def generate_transcript(severity, args, seed):
    persona, profile = PERSONAS[severity], PROFILES[severity]
    rng = random.Random(seed)
    n_turns = rng.randint(int(args.max_turns * 0.75), args.max_turns)
    n_turns -= n_turns % 2 == 0
    roles = plan_roles(n_turns, profile["p_patient"], rng)
    names = {"patient": persona["patient"], "caregiver": persona["caregiver"], "provider": persona["provider"]}

    texts, lines, fallback = [], [], []
    for start in range(0, n_turns, args.chunk_size):
        turns = list(range(start, min(start + args.chunk_size, n_turns)))
        found = request_turns(turns, roles, lines, persona, profile, args)
        for i in turns:
            if i not in found:
                fallback.append(i)
            texts.append(found.get(i, FALLBACK_TEXT[roles[i]]))
            lines.append(f"{names[roles[i]]} ({roles[i]}): {texts[-1]}")
        print(f"  {len(texts)}/{n_turns} turns")

    times = add_timing(roles, texts, profile, rng)
    df = pd.DataFrame({
        "turn": range(n_turns), "role": roles, "speaker": [names[r] for r in roles], "text": texts,
        "timestamp_seconds": [t[0] for t in times], "end_seconds": [t[1] for t in times],
    })
    meta = {
        "patient_id": persona["patient_id"], "severity": severity, "assumed_dsrs": profile["assumed_dsrs"], "persona": persona,
        "n_turns": n_turns, "role_share_pct": (df["role"].value_counts(normalize=True) * 100).round(1).to_dict(),
        "duration_seconds": float(df["end_seconds"].max()), "fallback_turns": fallback,
        "generation": {"model": args.model, "base_url": args.base_url, "temperature": args.temperature, "seed": seed},
    }
    return df, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate synthetic dementia clinic visit transcripts with an OpenAI-compatible LLM.")
    ap.add_argument("--severity", nargs="+", choices=[*PROFILES, "all"], default=["all"])
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--chunk-size", type=int, default=8, help="turns written per model call")
    ap.add_argument("--base-url", default="https://chat.dartmouth.edu/api", help="any OpenAI-compatible server")
    ap.add_argument("--model", default="gpt-oss-120b", help="matched against the server's model list (https://chat.dartmouth.edu/api/models)")
    ap.add_argument("--api-key-env", default="DARTMOUTH_CHAT_API_KEY", help="environment variable holding the API key")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=4000, help="leave room for the model's reasoning as well as the turns")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="synthetic_transcripts")
    args = ap.parse_args(argv)

    args.api_key = os.environ.get(args.api_key_env, "")
    if not args.api_key and "dartmouth" in args.base_url:
        raise SystemExit(f"set your Dartmouth Chat API key in the {args.api_key_env} environment variable")
    args.model = resolve_model(args)
    severities = list(PROFILES) if "all" in args.severity else args.severity

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for k, severity in enumerate(severities):
        print(f"\n{severity}")
        df, meta = generate_transcript(severity, args, args.seed + k)
        filename = f"{meta['patient_id']}_{severity}"
        df.to_csv(output_dir / f"{filename}.csv", index=False)
        (output_dir / f"{filename}_meta.json").write_text(json.dumps(meta, indent=2))
        shares = ", ".join(f"{r} {v}%" for r, v in sorted(meta["role_share_pct"].items()))
        print(f"{filename}: {meta['n_turns']} turns, {meta['duration_seconds'] / 60:.1f} min, {shares}, fallback turns {len(meta['fallback_turns'])}")


if __name__ == "__main__":
    main()
