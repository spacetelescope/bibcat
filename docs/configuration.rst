User Configuration and Data Filepaths
=====================================

There are three user environment variables to set:

- **BIBCAT_CONFIG_DIR**: a local path to your user configuration yaml file
- **BIBCAT_OPSDATA_DIR** : a local path to the directory of operational data in JSON format.
- **BIBCAT_DATA_DIR**: a local path to the directory of input data, e.g the input JSON files and full text
- **BIBCAT_OUTPUT_DIR**: a local path to a directory where the output of bibcat will be written, e.g. the output model and QA plots

If not set, all envvars will default to the user's home directory.  You can set these environment variables in your shell terminal, or in your shell config file, i.e. `.bashrc` or `.zshrc` file. For example,::

    export BIBCAT_CONFIG_DIR=/my/local/path/to/custom/config
    export BIBCAT_DATA_DIR=/my/local/path/to/input/data/dir
    export BIBCAT_OPSDATA_DIR=/my/local/path/to/operational/data/dir
    export BIBCAT_OUTPUT_DIR=/my/local/path/to/bibcat/output


All `bibcat` configuration is contained in a YAML configuration file, `bibcat_config.yaml` .  The default settings are located in `etc/bibcat_config.yaml`.  You don't modify this file directly.  To modify any of the settings, you do so through a custom user configuration file of the same name, placed in `$BIBCAT_CONFIG_DIR` or your home directory, mirroring the same default structure.  All user custom settings override the defaults.

For example, to change the name of the output model saved, within your user `$BIBCAT_CONFIG_DIR/bibcat_config.yaml`, set::

    output:
    name_model: my_new_model
