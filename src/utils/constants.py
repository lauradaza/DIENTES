ROOT = "/Path/to/dataset(s)/"

toothfairy = set(range(1, 49))
empty = set([19, 20, 29, 30, 39, 40])  # NA categories
toothfairy = list(toothfairy - empty)

TEMPLATE = {
    "1": toothfairy,
}

TASK_NAMES = {
    "1": "ToothFairy2",
}

REVERSE_TASK_NAMES = {v: k for k, v in TASK_NAMES.items()}

TASK_MODALITIES = {
    "1": "Cone Beam Computed Tomography",
}

ORGAN_NAMES = {
    1: "Lower Jawbone",
    2: "Upper Jawbone",
    3: "Lower Left Alveolar Canal",
    4: "Lower Right Alveolar Canal",
    5: "Upper Left Sinus",
    6: "Upper Right Sinus",
    7: "Pharynx",
    8: "Bridge",
    9: "Crown",
    10: "Implant",
    11: "Upper Right Central Incisor",
    12: "Upper Right Lateral Incisor",
    13: "Upper Right Canine",
    14: "Upper Right 1st Pre-molar",
    15: "Upper Right 2nd Pre-molar",
    16: "Upper Right 1st Molar",
    17: "Upper Right 2nd Molar",
    18: "Upper Right Wisdom Tooth",
    19: "NA1",
    20: "NA2",
    21: "Upper Left Central Incisor",
    22: "Upper Left Lateral Incisor",
    23: "Upper Left Canine",
    24: "Upper Left 1st Pre-molar",
    25: "Upper Left 2nd Pre-molar",
    26: "Upper Left 1st Molar",
    27: "Upper Left 2nd Molar",
    28: "Upper Left Wisdom Tooth",
    29: "NA3",
    30: "NA4",
    31: "Lower Left Central Incisor",
    32: "Lower Left Lateral Incisor",
    33: "Lower Left Canine",
    34: "Lower Left 1st Pre-molar",
    35: "Lower Left 2nd Pre-molar",
    36: "Lower Left 1st Molar",
    37: "Lower Left 2nd Molar",
    38: "Lower Left Wisdom Tooth",
    39: "NA5",
    40: "NA6",
    41: "Lower Right Central Incisor",
    42: "Lower Right Lateral Incisor",
    43: "Lower Right Canine",
    44: "Lower Right 1st Pre-molar",
    45: "Lower Right 2nd Pre-molar",
    46: "Lower Right 1st Molar",
    47: "Lower Right 2nd Molar",
    48: "Lower Right Wisdom Tooth",
}

NUM_CLASS = len(ORGAN_NAMES.keys())

MERGE_MAPPING = {
    "1": [(i, i) for i in toothfairy],
}
