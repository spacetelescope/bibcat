Input text data
===============

This ``bibcat/data/`` contains scripts to build the input data file, streamline the dataset, and partition the dataset into training, validation, and testing sets for the pretrained model method. The combined dataset is also used for the LLM model method.

Due to copyright and licensing issues, we are unable to provide the original full-text data. We assume that you have your own full text data in JSON format.

To construct the input dataset JSON file (``combined_data``), you need two sets of data: a set of full-text data and its corresponding classified label data. Both datasets should be in JSON format. We refer ``papertext`` to the ADS full text and ``papertrack`` to the MAST PaperTrack data.

- keys_papertext (from ADS provided full text file):

  ["bibcode", "abstract", "author", "bibstem", "identifier", "keyword",
  "keyword_norm", "page", "pub", "pub_raw", "pubdate", "title", "volume",
  "aff_canonical", "institution", "body"]

- keys for papertrack (from MAST Bibliography classification data):

  ["bibcode", "searches":["search_key","ignored"], "class_missions":["bibcode","papertype"]]

   where ``search_key`` refers to search mission names such as `HST`, or `Kepler`,
          ``ignored`` indicates that the paper is not related to the search mission, and
          ``papertype`` refers to its paper classification such as `science`, `datafinfluenced`, or `mention`.

- The keys of the combined input data are
  ["bibcode", "abstract", "author", "keyword", "keyword_norm", "pubdate", "title", "body",
  "class_missions", "is_ignored_mission"]

   where ``class_mssions`` refers to ``papertype`` for the search mission, and
         ``is_igored_mission`` indicates that the paper is not related to the search mission.
