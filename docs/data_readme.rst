Input text data
===============

The ``bibcat/data/`` directory contains scripts for preparing input datasets. This includes building the initial dataset, cleaning and streamlining it, and splitting it into training, validation, and test sets for use with the `pretrained model approach <https://bibcat.readthedocs.io/en/latest/pretrained.html>`_. The same processed data is also used in the `LLM-based method <https://bibcat.readthedocs.io/en/latest/llm.html>`_.

To construct the combined input dataset file (``combined_data*.json``), two JSON sources are required: full-text data and corresponding classified label data. We refer to these as:

- **papertext** – the full-text dataset
- **papertrack** – the classification labels from MAST PaperTrack

**Minimum required keys and structure for ``papertext``:**

.. code-block:: json

   [
     {
       "bibcode": "...",
       "abstract": "...",
       "pubdate": "...",
       "title": ["..."],
       "body": "..."
     }
   ]


**Required keys and structure for ``papertrack``:**

.. code-block:: json

   [
     {
       "bibcode": "...",
       "searches": [
         {
           "search_key": "...",
           "ignored": false
         }
       ],
       "class_missions": [
         {
           "bibcode": "...",
           "papertype": "..."
         }
       ]
     }
   ]

Here:

- ``search_key`` is search key,  *mast* or *library*)
- ``ignored`` is a boolean (*true* or *false*) which indicates whether the paper is unrelated to the mission,
- ``papertype`` is the classification label (e.g., *SCIENCE*, *DATA-INFLUENCED*, *MENTION*)

**Keys in ``required combined input dataset`` needed for the BibCat:**

.. code-block:: json

   [
     "bibcode",
     "abstract",
     "pubdate",
     "title",
     "body",
     "class_missions",
     "papertype"
   ]

**Keys in the final combined input dataset:**

.. code-block:: json

   [
     "bibcode",
     "abstract",
     "author",
     "keyword",
     "keyword_norm",
     "pubdate",
     "title",
     "body",
     "class_missions",
     "papertype",
     "is_ignored_<mission>"
   ]

- ``class_missions`` maps each paper to its classification(s) by mission.
- ``is_ignored_<mission>`` is a boolean flag indicating whether the paper is unrelated to the specified mission.
- ``mission`` is either *mast* or *libary*.

The example of the combined dataset JSON format is as follows.

.. code-block:: json

   {
     "bibcode": "3023Natur.111..123y",
     "abstract": "We report a newly discovered Type Ia supernova, SN 3023X.",
     "author": [
       "Lastname1, Firstname1",
       "Lastname2, Firstname2"
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
     "class_missions": {
       "HST": {
         "bibcode": "3023Natur.111..123y",
         "papertype": "SCIENCE"
       }
     },
     "is_ignored_library": false,
     "is_ignored_mast": true
   }
