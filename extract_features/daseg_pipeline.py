"""
Skyler Heininger

This pipeline uses Zelasko's daseg pipeline to perform dialogue act segmentation.

Note this saves DAs at the word level, converting utterance level to word level.
This does mean that start times, etc, are duplicated across.

This is actually a pretty old script, that has been adapted for a couple different exploratory projects.
As such, some pieces may not make too much sense. However, this is purely for extracting the dialogue acts.
Other scripts focuses on grouping dialogue acts into the DA-level transcript, or grouping into 
specific DA groups.

The modification for this project is primarily with data loading.
"""

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, pipeline

IGNORE_MAX_LENGTH = False

MODEL_NAME = "pzelasko/longformer-swda-nolower"


def get_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return -1


def load_pipeline():
    device = get_device()
    label = {0: "CUDA GPU", -1: "CPU"}.get(device, str(device))
    print(f"Using device: {label}", flush=True)
    pipe = pipeline("token-classification", model=MODEL_NAME, device=device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return pipe, tokenizer


def concatenate_words_to_length(words, max_length=512):
    if IGNORE_MAX_LENGTH:
        return [" ".join(map(str, words))]

    windows = []
    current_window = []
    current_length = 0

    for word in words:
        word = str(word)
        if current_length + len(word) + 1 > max_length:
            windows.append(" ".join(current_window))
            current_window = [word]
            current_length = len(word)
        else:
            current_window.append(word)
            current_length += len(word) + 1

    if current_window:
        windows.append(" ".join(current_window))

    return windows


def scrub_word(word):
    """
    Remove non-ASCII and other characters that crash daseg. Some of these
    were common in certain transcription software (namely WhisperX)
    """
    chars_to_remove = ["Ä", "Ġ", " ", "Š", "\n"]
    original_word = str(word).replace("â€", "")
    original_word = original_word.replace("ÄŠ", "")
    original_word = original_word.replace("Ä\x8a", "")
    original_word = re.sub(r"[^\x00-\x7F]+", "", original_word)
    return "".join(char for char in original_word if char not in chars_to_remove)


def clean_prediction(prediction, chars_to_remove=None):
    """
    Clean prediction artifacts from daseg's token output.
    """
    if chars_to_remove is None:
        chars_to_remove = ["Ä", "Ġ", " ", "Š", "\n"]

    original_word = prediction["word"].strip()
    original_word = str(original_word).replace("â€", "")
    original_word = original_word.replace("ÄŠ", "")
    original_word = original_word.replace("Ä\x8a", "")
    cleaned_word = "".join(char for char in original_word if char not in chars_to_remove)
    prediction["word"] = cleaned_word
    return prediction, cleaned_word


def align_predictions_with_words_using_word_endings(words, predictions):
    """
    Align token-level predictions back to word level using word endings:
    each word consumes predictions one at a time until the cleaned
    prediction text matches the end of that word.
    """
    complete_predictions = []
    for predictions_list in predictions:
        complete_predictions.extend(predictions_list)
    print(f"Total tokens to align: {len(complete_predictions)}", flush=True)

    aligned_predictions = []
    prediction_index = 0
    num_predictions = len(complete_predictions)

    for word in words:
        word = str(word)
        word_predictions = []
        cleaned_word = scrub_word(word)

        while prediction_index < num_predictions:
            prediction = complete_predictions[prediction_index]
            prediction, cleaned_pred = clean_prediction(prediction)

            if not cleaned_pred:
                prediction_index += 1
                continue

            word_predictions.append(prediction)
            prediction_index += 1

            if cleaned_word.endswith(cleaned_pred):
                break

        aligned_predictions.append(word_predictions if word_predictions else None)

    return aligned_predictions


def turn_df_to_word_df(df, col_with_text):
    return (
        df.assign(**{col_with_text: df[col_with_text].str.split(" ")})
        .explode(col_with_text)
        .loc[lambda d: d[col_with_text].notna() & (d[col_with_text].str.strip() != "")]
        .reset_index(drop=True)
    )


def pad_list_to_dataframe_length(data, lst):
    df_length = len(data)
    if len(lst) < df_length:
        lst = lst + [""] * (df_length - len(lst))
    elif len(lst) > df_length:
        lst = lst[:df_length]
    return lst


def load_transcript(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    if file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    if file_path.endswith(".tsv"):
        return pd.read_csv(file_path, delimiter="\t")
    raise ValueError(f"Unsupported file format: {file_path}. Use CSV, TSV, or XLSX.")


def process_file(file_path, output_dir, col_with_text, pipe, tokenizer):
    """
    Load one transcript, run daseg, align predictions to words, and save
    the word-level output CSV to output_dir. Returns True on success,
    False if the file had nothing usable to process.
    """
    print(f"Processing: {file_path}", flush=True)
    data = load_transcript(file_path)

    data = data.dropna(how="all")
    data = data[data[col_with_text].notna() & (data[col_with_text].astype(str).str.strip() != "")]
    data = data[~(data[col_with_text] == "â€”")]
    data[col_with_text] = data[col_with_text].astype(str).str.replace(r"=-{2,}", "", regex=True)
    data = data[data[col_with_text].str.strip() != ""]
    data[col_with_text] = data[col_with_text].replace({"—": " ", "–": " ", "-": " "}, regex=True)

    data_word_level = turn_df_to_word_df(data, col_with_text)
    words = [scrub_word(str(w)) for w in data_word_level[col_with_text]]

    if not words:
        print(f"No words found in {file_path} -- skipping.", flush=True)
        return False

    windows = concatenate_words_to_length(words)
    windows = [w for w in windows if w and w.strip()]
    if not windows:
        print("No windows produced -- skipping.", flush=True)
        return False

    print(f"{len(words)} words -> {len(windows)} window(s)", flush=True)

    dataset = Dataset.from_dict({col_with_text: windows})
    predictions = pipe(list(dataset[col_with_text]), batch_size=16)

    word_predictions = align_predictions_with_words_using_word_endings(words, predictions)
    print(f"Aligned {len(word_predictions)} words", flush=True)

    entities, raw_entities, scores, raw_scores, words_pred, chunks = [], [], [], [], [], []
    chunk_index = 0
    previous_raw_label = "I-"
    temp_predictions = []

    for pred_list in word_predictions:
        if pred_list is None:
            chunk_index += 1
            for lst in (entities, raw_entities, scores, raw_scores, words_pred, chunks):
                lst.append("")
            continue

        pred = pred_list[0]
        entity = pred["entity"]
        score = pred["score"]

        temp_word = "".join(p["word"] for p in pred_list)
        temp_predictions.append({
            "entity": entity, "raw_entity": entity,
            "score": score, "raw_score": score, "word": temp_word,
        })

        if entity != previous_raw_label and previous_raw_label != "I-":
            chunk_index += 1

        if entity != "I-":
            for tp in temp_predictions:
                if tp["entity"] == "I-":
                    tp["entity"] = entity
                    tp["score"] = score
            for tp in temp_predictions:
                entities.append(tp["entity"])
                raw_entities.append(tp["raw_entity"])
                raw_scores.append(tp["raw_score"])
                scores.append(tp["score"])
                chunks.append(chunk_index)
                words_pred.append(tp["word"])
            temp_predictions = []

        previous_raw_label = entity

    for tp in temp_predictions:
        entities.append(tp["entity"])
        raw_entities.append(tp["raw_entity"])
        raw_scores.append(tp["raw_score"])
        scores.append(tp["score"])
        chunks.append(chunk_index)
        words_pred.append(tp["word"])

    data_word_level["Pred_DA"] = pad_list_to_dataframe_length(data_word_level, raw_entities)
    data_word_level["Raw_Score"] = pad_list_to_dataframe_length(data_word_level, raw_scores)
    data_word_level["Proc_DA"] = pad_list_to_dataframe_length(data_word_level, entities)
    data_word_level["Score"] = pad_list_to_dataframe_length(data_word_level, scores)
    data_word_level["Words_Prediction"] = pad_list_to_dataframe_length(data_word_level, words_pred)
    data_word_level["DA_number"] = pad_list_to_dataframe_length(data_word_level, chunks)

    none_count = sum(1 for p in word_predictions if p is None)
    print(f"  None predictions: {none_count}", flush=True)

    stem = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{stem}_with_predictions.csv")
    data_word_level.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}", flush=True)
    return True


def run_directory(directory, output_dir, col_with_text, pipe, tokenizer):
    """
    Modified from older versions to work on the subdirectory structure of the voices transcripts.
    """
    directory = Path(directory)
    output_dir = Path(output_dir)

    input_files = sorted(
        p for p in list(directory.glob("**/*.csv"))
        + list(directory.glob("**/*.xlsx"))
        + list(directory.glob("**/*.tsv"))
        if not p.name.startswith("~$")
    )
    print(f"{len(input_files)} file(s) found under {directory} (recursively).")

    n_processed = 0
    n_skipped = 0
    for path in input_files:
        rel_dir = path.relative_to(directory).parent
        out_subdir = output_dir / rel_dir
        expected_output = out_subdir / f"{path.stem}_with_predictions.csv"

        if expected_output.exists():
            n_skipped += 1
            continue

        out_subdir.mkdir(parents=True, exist_ok=True)
        try:
            process_file(str(path), str(out_subdir), col_with_text, pipe, tokenizer)
            n_processed += 1
        except Exception as e:
            print(f"ERROR on {path}: {e}", flush=True)

    print(f"\nProcessed {n_processed} file(s), skipped {n_skipped} already-processed file(s).")


def main():
    parser = argparse.ArgumentParser(description="Run daseg (dialogue act segmentation) on transcript(s).")
    parser.add_argument("--transcript", default=None, help="Path to a single transcript file (CSV/TSV/XLSX)")
    parser.add_argument("--directory", default=None, help="Directory of transcript files to process in bulk")
    parser.add_argument("--output_dir", required=True, help="Where output files will be written")
    parser.add_argument("--col_with_text", required=True, help="Column in the transcript containing spoken text")
    args = parser.parse_args()

    if bool(args.transcript) == bool(args.directory):
        raise SystemExit("Specify exactly one of --transcript or --directory.")

    os.makedirs(args.output_dir, exist_ok=True)
    pipe, tokenizer = load_pipeline()

    start = time.time()
    if args.transcript:
        if not os.path.exists(args.transcript):
            raise FileNotFoundError(f"Transcript not found: {args.transcript}")
        process_file(args.transcript, args.output_dir, args.col_with_text, pipe, tokenizer)
    else:
        run_directory(args.directory, args.output_dir, args.col_with_text, pipe, tokenizer)

    print(f"\nDone in {time.time() - start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
