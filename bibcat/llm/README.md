

# LLM Prompting

Here we describe the `bibcat` options for extracting paper information using LLMs.  It currently only supports using the OpenAI API to prompt ``gpt`` models.  The default model used is `gpt4o-mini`, but can be customized using the ``config.llms.openai.model`` field.

To use this code you must have an OpenAI API key. Follow [these instructions](https://platform.openai.com/docs/quickstart) and set your new API key to a new `OPENAI_API_KEY` environment variable.


## Submitting a paper

The cli `bibcat run-gpt` submits a prompt with paper content.  The paper content can either be a local filepath on disk, or a bibcode or array list index from source dataset ``dataset_combined_all_2018-2023.json``.

The `run-gpt` command submits a single paper.  The `batch-submit` command
submits a list of files.  The list can either be specified one of two ways:  1.) individually, with the `-f` option, as a filename, bibcode, or index, or 2.) a file that contains a list of filenames, bibcodes, or indicies to use, with one line per entry.

```python
# submit a paper
bibcat run-gpt -f /Users/bcherinka/Downloads/2406.15083v1.pdf

# use a bibcode from the source dataset
bibcat run-gpt -b 2023Natur.616..266L

# submit with entry 101 from the papertrack source dataset
bibcat run-gpt -i 101

# batch submit a list of papers, using source index
bibcat batch-submit -f 101 -f 102 -f 103

# batch submit from a file of files
bibcat batch-submit -p papers_to_process.txt
```

## User Configuration
An example for the new ``llms`` section of the configuration.

```yaml
llms:
  user_prompt: "What is this dataset?"
  agent_prompt: null
  prompt_output_file: paper_output.json
  llm_user_prompt: llm_user_prompt.txt
  llm_agent_prompt: llm_agent_prompt.txt
  openai:
    model: gpt-4o-mini
    asst_id: null
    use_assistant: false
```

## User and Agent (System) Prompts

User and agent prompts to the LLM can be customized through the `bibcat` configuration file. User and Agent prompts can be customized by creating custom prompt files. Custom prompt files take precedence over any config settings or defaults.  A ``user`` prompt is the text you provide the LLM as the question to be answered, or the task to be performed.  An ``agent`` prompt is the set of instructions, behaviors, or personality you wish the LLM model to follow when responding to your prompt.

To create a new custom user or agent prompt file, create a new text file at the location of `$BIBCAT_DATA_DIR`.  The name of the file can be anything, but must be specified in your `config.llms.llm_user_prompt`, and `config.llms.llm_agent_prompt` fields.  These fields are the filename references.  The custom prompt file is preferred as it allows for the specification of larger, more complex prompt text and instructions.

If no custom prompt file is found, the code defaults to the prompts found in `config.llms.user_prompt` and ``config.llms.agent_prompt`` or the default agent prompt in `etc/bibcat_config.yaml`.


## Assistants

By default, ``bibcat`` submits a straight [chat completion](https://platform.openai.com/docs/guides/chat-completions/overview) to the LLM, using the given user and agent prompts.  Alternatively, you can set up and use an OpenAI [Assistant](https://platform.openai.com/docs/assistants/overview) with [file-search](https://platform.openai.com/docs/assistants/tools/file-search) capability, and submit your paper to your new Assistant.

List your available assistants with ``bibcat openai list_assistants``.  Create a new assistant with ``bibcat openi create_assistant``.  When creating an assistant, it is set up using your custom agent prompt.  You can see your new assistant at https://platform.openai.com/assistants, modify it, play with it, etc.  After you create an assistant, it will have an `assistant_id` or `asst_id`, e.g. ``asst_85Ay7apSmArPOMKF4iRY893b``.  Set this value to the `config.llms.openai.asst_id` field to instruct `bibcat` to use this agent.

To tell bibcat to use the assistant, set `config.llms.openai.use_assistant=true`.

When submitting a paper to an assistant, the file or text is first uploaded to the Assistant.  The Assistant creates a temporary vector store database, reads the file content, and embeds it into the database.  The Assistant will search the file via the vector store when responding to your user prompt.

The Assistant supports uploading PDF or JSON files.  `bibcat` will accept either a local PDF file, as well as a bibcode entry from our source dataset.  When specifying a paper from the source dataset, using a bibcode or array index, `bibcat` will write the JSON data out to a temporary JSON file for upload.  The temporary filename syntax is of the form `temp_xxxx_(bibcode).json`, prefixed by `temp_` and suffixed by the bibcode of the paper.


## Response Output

The output response from the LLM prompt is written to a file, specified by `config.llms.prompt_output_file`, e.g. "paper_output.json".  The response output is organized by the name of the file, or the bibcode of the paper.  Repeated prompts using the same paper will be appended to the entry for that paper.


