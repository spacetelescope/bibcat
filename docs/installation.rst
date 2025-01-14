Installation
============

.. note::

    ``bibcat`` is undergoing constant development. We encourage users to always update
    to the latest version. In general, it is good practice to install the development
    version following the instructions below as full released versions may lag behind.

Packages and Conda Environment
------------------------------

Required packages and versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- tensorflow-macos 2.15.1
- tensorflow-metal 1.1.0
- tensorflow 2.15.0
- tensorflow-hub 0.16.1
- tensorflow-text 2.15.0
- See more packages found in the conda evn file (envs/bibcat_py310.yml).

Conda env installation
^^^^^^^^^^^^^^^^^^^^^^

Change `env_name` below with whatever you want to name the environment.
- Download the conda installation yml file [here](envs/bibcat_py310.yml).
- In the terminal, run these commands.::

    conda env create -n env_name -f bibcat_py310.yml
    conda activate env_name
    python -m spacy download en_core_web_sm


Extra required pacakge for Apple M1/M2/M3 chip and GPU use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Apple Silicon chips, to utilize your GPU, you install `**tensorflow-macos**` and `**tensorflow-metal**`.  You can run::

    pip install tensorflow-macos tensorflow-metal

To verify if tensorflow is set up to utilize your GPU, do the following:::

    import tensorflow as tf
    tf.config.list_physical_devices('GPU')

You should see the following output.::

    [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

If the output is an empty list, you are not setup for GPU use.

Install `tensorflow-text`
^^^^^^^^^^^^^^^^^^^^^^^^^
- You need to install this package manually. Follow the instruction order below.

- To install *tensorflow-text*, the command *pip install -U "tensorflow-text"* **does not work** due to some package version conflict (as of sometime 2024, need to revisit). You need to download the latest release library compatible with your system from the `Tensorflow library link. <https://github.com/sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon/releases>`_; For instance, if you have MacOSX with python 3.10, download `this library. <https://github.com/sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon/releases/download/v2.15/tensorflow_text-2.15.0-cp310-cp310-macosx_11_0_arm64.whl>`_ Then,::

    pip install /path-to-download/tensorflow_text-2.15.0-cp310-cp310-macosx_11_0_arm64.whl


BibCat installation
-------------------

The `bibcat` directory contains the python package itself, installable via pip.::

    pip install -e .

pre-commit for development
--------------------------

`pre-commit <https://pre-commit.com/>`_ allows all collaborators push their commits compliant with the same set of lint and format rules in [pyproject.toml](pyproject.toml) by checking all files in the project at different stages of the git workflow. It runs commands specified in the [.pre-commit-config.yaml](.pre-commit-config.yaml) config file and runs checks before committing or pushing, to catch errors that would have caused a build failure before they reach CI.

Install pre-commit
^^^^^^^^^^^^^^^^^^
You will need to install `pre-commit` manually.::

    pip install pre-commit # if you haven't already installed the package

    pre-commit install # install default hooks `pre-commit`, `pre-push`, and `commit-msg` as specified in the config file.


Run pre-commit
^^^^^^^^^^^^^^
If this is your first time running, you should run the hooks against for all files and it will fix all files based on your setting.::

    pre-commit run --all-files

Update pre-commit
^^^^^^^^^^^^^^^^^^
Finally, you will need to update `pre-commit` regularly by running::

    pre-commit autoupdate

For other configuration options and more detailed information, check out at the `pre-commit <https://pre-commit.com/>`_ page.
