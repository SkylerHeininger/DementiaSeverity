"""
Skyler Heininger
For processing the severity files from categorical text to integers.
"""

import argparse
import difflib
import re
from pathlib import Path

import pandas as pd

ITEM_ORDER = [
    "memory",
    "speech_language",
    "recognition",
    "orientation_time",
    "orientation_place",
    "decisions",
    "social_community",
    "home_activities",
    "personal_care",
    "eating",
    "urination_bowels",
    "mobility",
]

ITEM_LABELS = {
    "memory": "Memory",
    "speech_language": "Speech and Language",
    "recognition": "Recognition of Family Members",
    "orientation_time": "Orientation to Time",
    "orientation_place": "Orientation to Place",
    "decisions": "Ability to Make Decisions",
    "social_community": "Social and Community Activity",
    "home_activities": "Home Activities and Responsibilities",
    "personal_care": "Personal Care - Cleanliness",
    "eating": "Eating",
    "urination_bowels": "Control of Urination and Bowels",
    "mobility": "Ability to Get From Place to Place",
}

ITEM_OPTIONS: dict[str, list[tuple[int, str]]] = {
    "memory": [
        (0, "Normal memory."),
        (1, "Occasionally forgets things that they were told recently. Does not cause many problems."),
        (2, "Mild consistent forgetfulness. Remembers recent events but often forgets parts."),
        (3, "Moderate memory loss. Worse for recent events. May not remember something you just told them. Causes problems with everyday activities."),
        (4, "Substantial memory loss. Quickly forgets recent or newly-learned things. Can only remember things that they have known for a long time."),
        (5, "Does not remember basic facts like the day of the week, when last meal was eaten or what the next meal will be."),
        (6, "Does not remember even the most basic things."),
    ],
    "speech_language": [
        (0, "Normal ability to talk and to understand others."),
        (1, "Sometimes cannot find a word, but able to carry on conversations."),
        (2, "Often forgets words. May use the wrong word in its place. Some trouble expressing thoughts and giving answers."),
        (3, "Usually answers questions using sentences but rarely starts a conversation."),
        (4, "Answers questions, but responses are often hard to understand or don't make sense. Usually able to follow simple instructions."),
        (5, "Speech often does not make sense. Can not answer questions or follow instructions."),
        (6, "Does not respond most of the time."),
    ],
    "recognition": [
        (0, "Normal - recognizes people and generally knows who they are."),
        (1, "Usually recognizes grandchildren, cousins or relatives who are not seen frequently but may not recall how they are related."),
        (2, "Usually does not recognize family members who are not seen frequently. Is often confused about how family members such as grandchildren, nieces, or nephews are related to them."),
        (3, "Sometimes does not recognize close family members or others who they see frequently. May not recognize their children, brothers, or sisters who are not seen on a regular basis."),
        (4, "Frequently does not recognize spouse or caregiver."),
        (5, "No recognition or awareness of the presence of others."),
    ],
    "orientation_time": [
        (0, "Normal awareness of time of day and day of week."),
        (1, "Some confusion about what time it is or what day of the week, but not severe enough to interfere with everyday activities."),
        (2, "Frequently confused about time of day."),
        (3, "Almost always confused about the time of day."),
        (4, "Seems completely unaware of time."),
    ],
    "orientation_place": [
        (0, "Normal awareness of where they are even in new places."),
        (1, "Sometimes disoriented in new places."),
        (2, "Frequently disoriented in new places."),
        (3, "Usually disoriented, even in familiar places. May forget that they are already at home."),
        (4, "Almost always confused about place."),
    ],
    "decisions": [
        (0, "Normal - as able to make decisions as before."),
        (1, "Only some difficulty making decisions that arise in day-to-day life."),
        (2, "Moderate difficulty. Gets confused when things get complicated or plans change."),
        (3, "Rarely makes any important decisions. Gets confused easily."),
        (4, "Not able to understand what is happening most of the time."),
    ],
    "social_community": [
        (0, "Normal - acts the same with people as before."),
        (1, "Only mild problems that are not really important, but clearly acts differently from previous years."),
        (2, "Can still take part in community activities without help. May appear normal to people who don't know them."),
        (3, "Often has trouble dealing with people outside the home without help from caregiver. Usually can participate in quiet home activities with friends. The problem is clear to anyone who sees them."),
        (4, "No longer takes part in any real way in activities at home involving other people. Can only deal with the primary caregiver."),
        (5, "Little or no response even to primary caregiver."),
    ],
    "home_activities": [
        (0, "Normal. No decline in ability to do things around the house."),
        (1, "Some problems with home activities. May have more trouble with money management (paying bills) and fixing things. Can still go to a store, cook or clean. Still watches TV or reads a newspaper with interest and understanding."),
        (2, "Makes mistakes with easy tasks like going to a store, cooking or cleaning. Losing interest in the newspaper, TV or radio. Often can't follow a long conversation on a single topic."),
        (3, "Not able to shop, cook or clean without a lot of help. Does not understand the newspaper or the TV. Cannot follow a conversation."),
        (4, "No longer does any home-based activities."),
    ],
    "personal_care": [
        (0, "Normal. Takes care of self as well as they used to."),
        (1, "Sometimes forgets to wash, shave, comb hair, or may dress in wrong type of clothes. Not as neat as they used to be."),
        (2, "Requires help with dressing, washing and personal grooming."),
        (3, "Totally dependent on help for personal care."),
    ],
    "eating": [
        (0, "Normal, does not need help in eating food that is served to them."),
        (1, "May need help cutting food or have trouble with some foods, but basically able to eat by themselves."),
        (2, "Generally able to feed themselves but may require some help. May lose interest during the meal."),
        (3, "Needs to be fed. May have trouble swallowing."),
    ],
    "urination_bowels": [
        (0, "Normal - does not have problems controlling urination or bowels except for physical problems."),
        (1, "Rarely fails to control urination (generally less than one accident per month)."),
        (2, "Occasional failure to control urination (about once a week or less)."),
        (3, "Frequently fails to control urination (more than once a week)."),
        (4, "Generally fails to control urination and frequently can not control bowels."),
    ],
    "mobility": [
        (0, "Normal, able to get around on their own. (May have physical problems that require a cane or walker)."),
        (1, "Sometimes gets confused when driving or taking public transportation, especially in new places. Able to walk places alone."),
        (2, "Cannot drive or take public transportation alone, even in familiar places. Can walk alone outside for short distances. Might get lost if walking too far from home."),
        (3, "Cannot be left outside alone. Can get around the house without getting lost or confused."),
        (4, "Gets confused and needs help finding their way around the house."),
        (5, "Almost always in a bed or chair. May be able to walk a few steps with help, but lacks sense of direction."),
        (6, "Always in bed. Unable to sit or stand."),
    ],
}

SEVERITY_CUTOFFS = [
    (18, "normal_to_mild"),
    (36, "moderate"),
    (54, "severe"),
]

MATCH_THRESHOLD = 0.6


def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def coverage_score(response_tokens, option_tokens):
    if not response_tokens:
        return 0.0
    option_set = set(option_tokens)
    matched = sum(1 for t in response_tokens if t in option_set)
    return matched / len(response_tokens)


def match_response(response, options):
    if not response or not response.strip():
        return None, None, 0.0

    response_norm = normalize(response)

    for score, text in options:
        if response_norm == normalize(text):
            return score, text, 1.0

    response_tokens = tokenize(response)
    scored = []
    for score, text in options:
        cov = coverage_score(response_tokens, tokenize(text))
        ratio = difflib.SequenceMatcher(None, response_norm, normalize(text)).ratio()
        scored.append((cov, ratio, score, text))

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best_cov, best_ratio, best_score, best_text = scored[0]

    if best_cov >= MATCH_THRESHOLD:
        return best_score, best_text, best_cov
    return None, best_text, best_cov


def severity_category(total):
    for cutoff, label in SEVERITY_CUTOFFS:
        if total <= cutoff:
            return label
    return "severe"


def cell_to_str(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def read_xlsx(path):
    df = pd.read_excel(path, engine="openpyxl")
    fieldnames = [str(c) for c in df.columns]
    rows = [{k: cell_to_str(v) for k, v in record.items()} for record in df.to_dict(orient="records")]
    return fieldnames, rows


def read_csv_file(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    fieldnames = [str(c) for c in df.columns]
    rows = df.to_dict(orient="records")
    return fieldnames, rows


def read_table(path):
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return read_xlsx(path)
    if suffix == ".csv":
        return read_csv_file(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def write_xlsx(path, fieldnames, rows):
    df = pd.DataFrame(rows, columns=fieldnames)
    df.to_excel(path, index=False, engine="openpyxl")


def write_csv_file(path, fieldnames, rows):
    df = pd.DataFrame(rows, columns=fieldnames)
    df.to_csv(path, index=False)


def write_table(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        write_xlsx(path, fieldnames, rows)
    else:
        write_csv_file(path, fieldnames, rows)


def find_record_id_column(fieldnames):
    for name in fieldnames:
        if re.sub(r"[\s_]", "", name.lower()) == "recordid":
            return name
    return None


def find_memory_column_index(fieldnames):
    for i, name in enumerate(fieldnames):
        if "memory" in name.lower():
            return i
    return None


def detect_item_columns(fieldnames, override):
    if override is not None:
        if len(override) != len(ITEM_ORDER):
            raise ValueError(f"--item-columns needs exactly {len(ITEM_ORDER)} column names, got {len(override)}")
        return dict(zip(ITEM_ORDER, override))

    memory_idx = find_memory_column_index(fieldnames)
    if memory_idx is None:
        raise ValueError("Could not find a column containing 'memory' to anchor item detection.")

    if memory_idx + len(ITEM_ORDER) > len(fieldnames):
        raise ValueError(f"Found 'Memory' at column {memory_idx}, but there aren't {len(ITEM_ORDER)} columns after it. Use --item-columns to specify explicitly.")

    columns = fieldnames[memory_idx:memory_idx + len(ITEM_ORDER)]
    return dict(zip(ITEM_ORDER, columns))


def strip_record_id_suffix(record_id):
    return re.sub(r"_D$", "", record_id.strip(), flags=re.IGNORECASE)


def find_matching_transcript(record_id_clean, transcripts_dir):
    matches = [
        str(p.relative_to(transcripts_dir)) for p in transcripts_dir.glob("**/*")
        if p.is_file() and p.name.lower().startswith(record_id_clean.lower())
    ]
    return matches


def process_file(path, transcripts_dir, override):
    fieldnames, rows = read_table(path)

    record_id_col = find_record_id_column(fieldnames)
    if record_id_col is None:
        raise ValueError(f"Could not find a 'Record ID' column in {path}")

    item_columns = detect_item_columns(fieldnames, override)
    print(f"  detected item columns: {', '.join(item_columns[item] for item in ITEM_ORDER)}")

    appended_fields = ["record_id_clean"] + \
        [f"{item}_score" for item in ITEM_ORDER] + \
        ["total_score", "severity_category"] + \
        (["matched_transcript"] if transcripts_dir else [])

    out_rows = []
    n_incomplete = 0
    n_incomplete_with_transcript = 0
    n_flagged = 0

    for row in rows:
        record_id_raw = row.get(record_id_col, "")
        record_id_clean = strip_record_id_suffix(record_id_raw)

        out_row = dict(row)
        out_row["record_id_clean"] = record_id_clean

        has_transcript = False
        if transcripts_dir:
            matches = find_matching_transcript(record_id_clean, transcripts_dir)
            out_row["matched_transcript"] = ";".join(matches)
            has_transcript = len(matches) > 0

        scores = {}
        flagged_this_row = False

        for item in ITEM_ORDER:
            raw_response = row.get(item_columns[item], "")
            score, matched_text, cov = match_response(raw_response, ITEM_OPTIONS[item])
            out_row[f"{item}_score"] = score if score is not None else ""
            scores[item] = score
            if score is None and raw_response.strip():
                flagged_this_row = True
            elif score is not None and cov < 0.9:
                flagged_this_row = True

        valid_scores = [s for s in scores.values() if s is not None]
        if len(valid_scores) == len(ITEM_ORDER):
            total = sum(valid_scores)
            out_row["total_score"] = total
            out_row["severity_category"] = severity_category(total)
        else:
            out_row["total_score"] = ""
            out_row["severity_category"] = ""
            n_incomplete += 1
            if has_transcript:
                n_incomplete_with_transcript += 1

        if flagged_this_row:
            n_flagged += 1

        out_rows.append(out_row)

    out_fieldnames = fieldnames + appended_fields
    return out_rows, out_fieldnames, n_incomplete, n_incomplete_with_transcript, n_flagged


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsrs-dir", default=None,
                     help="Directory containing one or more raw DSRS export CSVs. Processes every file recursively")
    ap.add_argument("--dsrs-file", default=None,
                     help="Process exactly this one DSRS file, ignoring everything else that might "
                          "be sitting in the same directory. Use this or --dsrs-dir, not both.")
    ap.add_argument("--output-subdir", default="processed", 
                    help="Subdirectory (created under the file's own parent dir, or --dsrs-dir) "
                          "to write the processed output into")
    ap.add_argument("--transcripts-dir", default=None,
                     help="If given, match each record to its transcript file under this directory")
    ap.add_argument("--item-columns", default=None,
                     help="Comma-separated list of the 12 item column names, in DSRS order, "
                          "if auto-detection picks the wrong columns. Applies to every file.")
    args = ap.parse_args()

    if bool(args.dsrs_dir) == bool(args.dsrs_file):
        raise SystemExit("Specify exactly one of --dsrs-dir or --dsrs-file.")

    override = args.item_columns.split(",") if args.item_columns else None
    transcripts_dir = Path(args.transcripts_dir) if args.transcripts_dir else None

    if args.dsrs_file:
        dsrs_file = Path(args.dsrs_file)
        if not dsrs_file.exists():
            raise SystemExit(f"{dsrs_file} does not exist")
        dsrs_dir = dsrs_file.parent
        output_dir = dsrs_dir / args.output_subdir
        csv_files = [dsrs_file]
    else:
        dsrs_dir = Path(args.dsrs_dir)
        output_dir = dsrs_dir / args.output_subdir
        csv_files = sorted(
            p for p in list(dsrs_dir.glob("**/*.xlsx")) + list(dsrs_dir.glob("**/*.csv"))
            if args.output_subdir not in p.relative_to(dsrs_dir).parts
            and not p.name.startswith("~$")
        )
        if not csv_files:
            raise SystemExit(f"No .xlsx or .csv files found under {dsrs_dir}")

    print(f"{len(csv_files)} DSRS file(s) to process {'under ' + str(dsrs_dir) if args.dsrs_dir else ''}")

    total_incomplete = 0
    total_incomplete_with_transcript = 0
    total_flagged = 0
    total_records = 0

    for i, path in enumerate(csv_files, start=1):
        rel_path = path.relative_to(dsrs_dir)
        print(f"\n[{i}/{len(csv_files)}] {rel_path}")
        try:
            out_rows, out_fieldnames, n_incomplete, n_incomplete_with_transcript, n_flagged = \
                process_file(path, transcripts_dir, override)
        except ValueError as e:
            print(f"Skipped: {e}")
            continue

        out_path = output_dir / rel_path
        write_table(out_path, out_fieldnames, out_rows)

        print(f"{len(out_rows)} record(s); {n_incomplete} incomplete; {n_flagged} flagged")
        if transcripts_dir:
                    print(f"{total_incomplete_with_transcript} of those {total_incomplete} incomplete record(s) have a matched transcript, just DSRS is incomplete")

        print(f"wrote {out_path}")

        total_records += len(out_rows)
        total_incomplete += n_incomplete
        total_incomplete_with_transcript += n_incomplete_with_transcript
        total_flagged += n_flagged

    print(f"Processed {total_records} records across {len(csv_files)} files.")
    print(f"{total_incomplete} record(s) have at least one unscoreable item.")
    if transcripts_dir:
        print(f"{total_incomplete_with_transcript} of those {total_incomplete} incomplete record(s) have a matched transcript, just DSRS is incomplete")
    print(f"Output written to {output_dir}")


if __name__ == "__main__":
    main()
