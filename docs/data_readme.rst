Input text data
===============

The ``bibcat/data/`` directory contains scripts for preparing the input dataset. This includes building the initial data file, streamlining the dataset, and partitioning it into training, validation, and test sets for `the pretrained model approach <pretrained.md>`_. The processed dataset is also used for `the LLM-based method <llm.md>`_.

To construct the input dataset JSON file (*combined_data\*.json*), two sources are required: full-text data and the corresponding classified label data, both in JSON format. We refer to the full-text dataset as **papertext** and the label dataset as **papertrack** (MAST PaperTrack classifications).

- Minimal content needed for **papertext**:

.. code-block:: python

  ["bibcode", "abstract", "pubdate", "title", "body"]

- The keys for **papertrack**:

.. code-block:: python

  ["bibcode", "searches":["search_key","ignored"], "class_missions":["bibcode","papertype"]]

Here, *search_key* refers to the mission name used for searching (e.g., HST, Kepler), *ignored* indicates that the paper is unrelated to the specified mission, and papertype specifies the paper’s classification, such as *SCIENCE*, *DATA-INFLUENCED*, or *MENTION*.

- The keys of the combined input data are

.. code-block:: python

  ["bibcode", "abstract", "author", "keyword", "keyword_norm", "pubdate", "title", "body", "class_missions", "papertype", "is_ignored_<mission>"]

Here, ``class_missions`` refers to the ``papertype`` classification for the search mission, and ``is_ignored_<mission>`` indicates that the paper is unrelated to the search mission—whether from **mast** missions or **library** flagship missions.
``

However, the requried metadata needed for the BibCat are as follows.

.. code-block:: python

  ["bibcode", "abstract", "pubdate", "title", "body", "class_missions", "papertype" ]

The example of the combined dataset JSON format is as follows.

.. code-block:: JSON

  {
    "bibcode": "3023Natur.111..123y",
    "abstract": "We report a newly discovered Type Ia supernova,SN 3023X.",
    "author": [
      "Lastname1, Firstname1",
      "Lastname2, Firstname2",
    ],
    "keyword": [
      "Astrophysics - High Energy Astrophysical Phenomena"
    ],
    "keyword_norm": [
      "-"
    ],
    "pubdate": "3023-12-00",
    "title": [
      "A discovery of a new peculiar Type Ia supernovae"
    ],
    "body": "We report the HST observation of SN 3023X, a Type Ia supernova exhibiting unusually slow decline rates and strong carbon absorption features pre-maximum—traits inconsistent with canonical models. Located in a passive elliptical galaxy at z = 0.034, its peak luminosity was 0.7 mag fainter than normal SNe Ia. Spectroscopic evolution suggests incomplete detonation or a hybrid progenitor. The anomaly challenges the standard candle assumption and may represent a new subclass. Continued photometric and spectroscopic monitoring is underway. SN 3023X offers a rare window into the diversity of thermonuclear explosions.",
    "class_missions": {     "HST": {
        "bibcode": "3023Natur.111..123y",
        "papertype": "SCIENCE"
      }
    },
    "is_ignored_library": false,
    "is_ignored_mast": true
  },
