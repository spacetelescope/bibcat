Input text data
===============

This ``bibcat/data/`` contains scripts to build the input text data file, streamline the dataset, and partition the dataset into training, validation, and testing sets.

Due to copyright and licensing issues, we are unable to provide the original full-text data. We assume that you have your own text file in JSON format.

To construct the input text JSON file, you need two sets of data: a full-text dataset and its corresponding classified label data. For the MAST bibliography, we used the MAST Papertrack DB data for paper classification (with a label, papertype, per bibcode) and the ADS full-text data (a full text per bibcode). Both datasets should be in JSON format.

The metadata keys for the MAST Papertrack data, the ADS full-text data, and the combined data are as follows. However, we only need the following keys from the ADS text metadata to construct the final input data: ["abstract", "author", "bibcode", "body", "keyword", "keyword_norm", "pubdate", "title"].

- keys_papertext (from ADS):

  ["bibcode", "abstract", "author", "bibstem", "identifier", "keyword",
  "keyword_norm", "page", "pub", "pub_raw", "pubdate", "title", "volume",
  "aff_canonical", "institution", "body"]
- keys for papertrack (from MAST):
-
  ["bibcode", "searches":["search_key","ignored"], "class_missions":["bibcode","papertype"]]

   where ``search_key`` refers to search mission names such as `HST`, or `Kepler`,
          ``ignored`` indicates that the paper is not related to the search mission, and
          ``papertype`` refers to its paper classification such as `science`, `datafinfluenced`, or `mention`.
- The combined input data:
  ["bibcode", "abstract", "author", "keyword", "keyword_norm", "pubdate", "title", "body",
  "class_missions", "is_ignored_mission"]

   where ``class_mssions`` refers to ``papertype`` for the search mission, and
         ``is_igored_mission`` indicates that the paper is not related to the search mission.
