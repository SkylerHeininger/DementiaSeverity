EXTENDED_DA_GROUPS = {
    "questions": [
        "Wh-Question", "Yes-No-Question", "Open-Question", "Or-Clause",
        "Declarative-Yes-No-Question", "Rhetorical-Questions",
        "Tag-Question", "Declarative-Wh-Question",
    ],
    "answers": [
        "Yes-answers", "No-answers",
        "Affirmative-non-yes-answers", "Other-answers",
        "Negative-non-no-answers", "Dispreferred-answers", "Reject",
        "Response-Acknowledgement", "Hold-before-answer-agreement",
        "Agree-Accept",
    ],
    "statements": ["Statement-non-opinion", "Statement-opinion"],
    "backchannel": [
        "Acknowledge-Backchannel", "Backchannel-in-question-form",
        "Non-verbal", "Signal-non-understanding",
    ],
    "hedge": ["Hedge"],
    "social_ritual": [
        "Conventional-opening", "Conventional-closing",
        "Thanking", "Apology", "Appreciation", "Downplayer",
        "3rd-party-talk",
    ],
    "reformulation": ["Summarize/reformulate", "Repeat-phrase"],
    "elaboration": ["Collaborative-Completion", "Quotation"],
    "action": ["Action-directive", "Offers-Options-Commits"],
    "unintelligible": ["Uninterpretable", "Other"],
}
DA_TO_GROUP = {da: group for group, members in EXTENDED_DA_GROUPS.items() for da in members}
GROUP_ORDER = sorted(EXTENDED_DA_GROUPS) + ["other"]


def map_da_to_group(label):
    return DA_TO_GROUP.get(str(label).strip(), "other")
