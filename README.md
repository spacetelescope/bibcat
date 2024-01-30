# Bibcat
-----------------------------------------------------------------------------

Bibcat classifies astronomical journal papers into multiple paper categories. The primary categories are "science", "mention", "data-influenced", and "ignore".
## Setup
### Input JSON file
Download several data files (the ADS full text file and the papertrack file) to create models for training or combined fulltext dataset files for the input text. These files can be accessed only by authorized users. Downloading the files requires a single sign-on.
Save the files outside the `bibcat` folder on your local computer, and you will set up the paths to the files. See more details in **Set paths for the data files** below.

- The combined papers+classification JSON file ([dataset_combined_all_2018-2021.json](https://stsci.app.box.com/file/1380606268376))
- The papertrack export JSON file ([papertrack_export_2023-11-06.json](https://stsci.box.com/s/zadlr8dixw8706o9k9smlxdk99yohw4d))
- ADS fulltext data file ([ST_Request2018-2021.json](https://stsci.box.com/s/cl3yg5mxqcz484w0iptwbl9b43h4tk1g))

Note that other JSON files (extracted from 2018-2023) include paper track data and full texts later than 2021. If you like to test them for your work, feel free to do so.

### Set paths for the data files
In `bibcat/config.py`, you will need to set several paths as follow.

#### The combined data set used for training models.
Set the combined data file path outside the `bibcat` folder, which should not be git-committed to the repo.
- `path_json = "/path/to/dataset_combined_all_2018-2021.json"`

#### When testing with pytest or unittest
In `tests/test_bibcat.py`
- `filepath_input = "/path/to/the/dataset"`
- `path_papertrack = os.path.join(filepath_input, "papertrack_export_2023-11-06.json")`
- `path_papertext = os.path.join(filepath_input, "ST_Request2018-2021.json")`
- Note that you need to set up the input file path outside the `bibcat` folder, which should not be git-committed to the repo.

## Quick start

- First, set the variables `path_json` (JSON file location path)  and `name_model` (Model name of your choice to save or load) in `bibcat/config.py`.
- Next, run `create_model.py` to create a training model.  
- Then, run `classify_papers.py` to classify papers. It will produce some evaluation diagnostics such as a confusion matrix in the `output/` directory.



License
-------

This project is Copyright (c) Jamila Pegues, Jinmi Yoon, Brian Cherinka and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Openastronomy packaging guide <https://github.com/OpenAstronomy/packaging-guide>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.

Contributing
------------

We love contributions! bibcat is open source,
built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
bibcat based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.
