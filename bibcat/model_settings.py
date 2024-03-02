"""
:title: model_settings.py

This module allows users to change the parameters required to build and train ML models.
"""

# num_papers: Set None to use all available papers in external dataset
# Note: If set to integer, final paper count might be a little more than
# target num_papers given
num_papers = 100

# Partition fraction of training, validation, testing dataset
fraction_TVT = [0.8, 0.1, 0.1]

# mode_TVT: "uniform" = all training datasets will have the same number of entries from fraction_TVT
# "available" = all training datasets will use full fraction
mode_TVT = "uniform"

seed_TVT = 10  # Random seed for generating train vs valid. vs test datasets
seed_ML = 8  # Random seed for ML model
# mode_modif: Mode to use for text processing and generating
# possible modes: any combination from "skim", "trim", and "anon" or "none"
mode_modif = "skim_anon"

# buffer: the number of sentences to include within paragraph around
# each sentence with target terms
buffer = 0
