# Dementia Severity Prediction

Skyler Heininger and Forrest Leung

## Data 

There are two forms of data, being the transcripts, and DSRS surveys. The transcripts are turn-speaker level, meaning each row is a turn that a speaker took in the conversation. The rows for this are as given:

```
turn_index | speaker | role | timestamp | timestamp_seconds | word_count | text
```

The DSRS surveys were given with the raw responses, meaning each column had the raw-text selection for that form field. This is relatively easy to handle, just needed to be converted to numberical outputs for each one. `process_severity/process.py` handles this piece. This folder also contains some basic prediction pieces. 

`speaker_roles/speaking_time.py` 


## Structural Feature Processing

We use the model, DASeg, made by Żelasko et al. The basic pipeline, for pulling these features is found in `extract_features/daseg_pipeline.py`. This provides word-level transcripts with matched dialogue acts. For the most part, this is the most basic level of dialogue acts. Processing scripts can then combine the word-level approaches by grouping on the `DA_number` col in the csvs.


## Structural Feature Modeling

Depending on the method, we either one-hot encode the speaker and dialogue act label, or simply pass in different channels. We also group-down the dialogue acts into more discernable groups. For example, the eight question groups, which individually can be quite rare, into one larger questions class. 

We use elasticnet on session-level features (speaker, DA groups) and a CNN to extract and predict sequences of DAs. See the cnn and elasticnet files within predict_severity dir.

## Saved models

models/ contains the object files of models, used in the interactive app.

## Interactive Component

See the readme within interactive folder for instructions.

You should make a virtual or conda env and install requirements using `pip install -r requirements.txt`

TAs will not be able to run anything other than the interactive component. All data used in this project contains personal health information, which is protected under HIPAA law and under Dartmouth IRB.
