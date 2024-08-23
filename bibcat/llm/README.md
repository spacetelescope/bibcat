

# LLM Prompting

Here we describe the `bibcat` options for extracting paper information using LLMs.  It currently only supports using the OpenAI API to prompt ``gpt`` models.  The default model used is `gpt4o-mini`, but can be customized using the ``config.llms.openai.model`` field.

This code requires the `openai` python package.  Install it with `pip install openai`.

To use this code you must have an OpenAI API key. Follow [these instructions](https://platform.openai.com/docs/quickstart) and set your new API key to a new `OPENAI_API_KEY` environment variable.


## Quick Start

Here is a quick start guide.

1. Follow the [Setup](../../README.md#setup) instructions from the main README
2. Select a paper bibcode or index
   1. run: `bibcat run-gpt -b "2023MNRAS.521..497M"`
   2. or: `bibcat run-gpt -i 50`
3. Check the response output `paper_output.json` file.

You should see something like `INFO - Output: {'HST': ['MENTION', 0.8], 'JWST': ['SCIENCE', 0.9]}`. See [Response Output](#response-output) for details about the response output.  [Submitting a paper](#submitting-a-paper) describes the command for sending papers to OpenAI. For customizing and trialing new gpt prompts, see [User Configuration](#user-configuration) and [User and Agent (System) Prompts](#user-and-agent-system-prompts).


## Submitting a paper

The cli `bibcat run-gpt` submits a prompt with paper content.  The paper content can either be a local filepath on disk, or a bibcode or array list index from source dataset ``dataset_combined_all_2018-2023.json``.  When specifying a bibcode or list array index, the paper data is pulled from the source JSON dataset.

The `run-gpt` command submits a single paper.
```python
# submit a paper
bibcat run-gpt -f /Users/bcherinka/Downloads/2406.15083v1.pdf

# use a bibcode from the source dataset
bibcat run-gpt -b 2023Natur.616..266L

# submit with entry 101 from the papertrack source dataset
bibcat run-gpt -i 101
```

The `run-gpt-batch` command submits a list of files.  The list can either be specified one of two ways:  1.) individually, with the `-f` option, as a filename, bibcode, or index, or 2.) a file that contains a list of filenames, bibcodes, or indicies to use, with one line per entry.

```python
# batch submit a list of papers, using source index
bibcat run-gpt-batch -f 101 -f 102 -f 103

# batch submit from a file of files
bibcat run-gpt-batch -p papers_to_process.txt
```

where `papers_to_process.txt` might look like
```txt
2023Natur.616..266L
2023ApJ...946L..13F
2023ApJS..265....5H
2023ApJ...954...31C
```

### Running within Python

To run within a Python environment, use the `classify_paper` function:
```python

from bibcat.llm.openai import classify_paper

# classify paper 200 from the dataset
classify_paper(index=200)
```
which gives output
```bash
Loading source dataset: /Users/bcherinka/Work/stsci/bibcat_data/dataset_combined_all_2018-2023.json
2024-08-22 15:49:50,634 - bibcat.llm.openai - INFO - Using paper bibcode: 2023MNRAS.518..456D
2024-08-22 15:49:56,781 - bibcat.llm.openai - INFO - Output: {'HST': ['MENTION', 0.7], 'JWST': ['SCIENCE', 0.9]}
```

## User Configuration
An example for the new ``llms`` section of the configuration.

```yaml
llms:
  user_prompt: null
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

User and agent prompts to the LLM can be customized through the `bibcat` configuration file, either via simple string prompts, specified with `config.llms.[role]_prompt` or via custom prompt files, specified with `config.llms.llm_[role]_prompt`.  Use the direct string prompts for simple prompts, like 'How old is the Earth?'.  For longer, or more complex, prompts, use the custom prompt files.  Custom prompt files take precedence over any config settings or defaults.  A ``user`` prompt is the text you provide the LLM as the question to be answered, or the task to be performed.  An ``agent`` prompt is the set of instructions, behaviors, or personality you wish the LLM model to follow when responding to your prompt.

To create a new custom user or agent prompt file, create a new text file at the location of `$BIBCAT_DATA_DIR`.  The name of the file can be anything, but must be specified in your `config.llms.llm_user_prompt`, and `config.llms.llm_agent_prompt` fields.  These fields are the filename references.  The custom prompt file is preferred as it allows for the specification of larger, more complex prompt text and instructions.

If no custom prompt file is found, the code defaults to the prompts found in `config.llms.user_prompt` and ``config.llms.agent_prompt`` or the default agent prompt in `etc/bibcat_config.yaml`.

### Testing New Prompts

When no custom config prompts or prompt files are found, the default user prompt (`etc/default_user_prompt.txt`) and system prompt (`etc/default_agent_prompt.txt`) is used.  There are three ways to change the default prompts used, to create custom prompts for trial and testing.

**Note:** The following are silly examples exploiting the use of `bibcat`, and are not indicative of package use.

#### Option 1 - Via Config

- For simple prompts, you can modify your config prompts directly:
```
llms:
  user_prompt: How large can a Beluga whale grow?
  agent_prompt: You are an professional expert on whales.  You are also witty and always respond with a clever pun.
  prompt_output_file: paper_output.json
  llm_user_prompt: null
  llm_agent_prompt: null
```

- Running `bibcat run-gpt -i 0 -v` in verbose mode produces:

```bash
Loading source dataset: /Users/bcherinka/Work/stsci/bibcat_data/dataset_combined_all_2018-2023.json
INFO - Using paper bibcode: 2023Natur.616..266L
WARNING - Error in prompt JSON response. Cannot convert output.
INFO - Agent Prompt: You are an professional expert on whales.  You are also witty and always respond with a clever pun.
INFO - User Prompt: How large can a Beluga whale grow?
INFO - Original Prompt Response: Beluga whales can grow to about 13 to 20 feet long, but some rare specimens can reach up to 24 feet! It seems the belugas really know how to make a "splash" when it comes to size!
INFO - Output: {'error': 'No JSON content found in response'}
```

#### Option 2 - Via Prompt File

For more complex prompts, specify them in custom files at `$BIBCAT_DATA_DIR`.

- Create a new text file, `my_user_prompt.txt` with the following content:
```txt
Which type of whale appeared in a 80's science fiction movie?
```
- Create a new text file, `my_agent_prompt.txt` with the following content:
```txt
You are an professional expert on whales. You are also witty and always respond with a clever pun. *Always* format your
response as valid JSON, with the following example as a guide:
"```json
{{
 "whale": "Blue",
 "response": "The Blue whale weighs up to 199 tons.",
 "source: "https://en.wikipedia.org/wiki/Blue_whale"
}}
```"
*Always* include a "source" field, which is real url link to your source of information
```
- Modify your config file as follows:
```
llms:
  user_prompt: null
  agent_prompt: null
  prompt_output_file: paper_output.json
  llm_user_prompt: my_user_prompt.txt
  llm_agent_prompt: my_agent_prompt.txt
```

- Running `bibcat run-gpt -i 0` without verbosity produces:
```bash
Loading source dataset: /Users/bcherinka/Work/stsci/bibcat_data/dataset_combined_all_2018-2023.json
INFO - Using paper bibcode: 2023Natur.616..266L
WARNING - Error converting output to classification format: too many values to unpack (expected 2)
INFO - Output: {'whale': 'Humpback', 'response': "The Humpback whale is the star of 'Star Trek IV: The Voyage Home'. Talk about a cinematic whale ready for a 'fin'-tastic adventure!", 'source': 'https://en.wikipedia.org/wiki/Star_Trek_IV:_The_Voyage_Home'}
```

#### Option 3 - Toggle via CLI

If you have multiple custom prompt files, you can quickly switch between them using the `-u` or `-a` flags on `run-gpt`, for `user`, and `agent` prompts, respectively, without modifying your config file.

- Create a new text file, `my_user_prompt2.txt` with the following content:
```txt
What color was the whale that Ahab hated?
```

- Running `bibcat run-gpt -i 0 -u my_user_prompt2.txt`, produces:
```bash
Loading source dataset: /Users/bcherinka/Work/stsci/bibcat_data/dataset_combined_all_2018-2023.json
INFO - Using paper bibcode: 2023Natur.616..266L
WARNING - Error converting output to classification format: too many values to unpack (expected 2)
INFO - Output: {'whale': 'White', 'response': "Captain Ahab had a 'whale' of a problem with Moby Dick, the infamous white whale!", 'source': 'https://en.wikipedia.org/wiki/Moby-Dick'}
```

## Assistants

By default, ``bibcat`` submits a straight [chat completion](https://platform.openai.com/docs/guides/chat-completions/overview) to the LLM, using the given user and agent prompts.  Alternatively, you can set up and use an OpenAI [Assistant](https://platform.openai.com/docs/assistants/overview) with [file-search](https://platform.openai.com/docs/assistants/tools/file-search) capability, and submit your paper to your new Assistant.

List your available assistants with ``bibcat openai list_assistants``.  Create a new assistant with ``bibcat openi create_assistant``.  When creating an assistant, it is set up using your custom agent prompt.  You can see your new assistant at https://platform.openai.com/assistants, modify it, play with it, etc.  After you create an assistant, it will have an `assistant_id` or `asst_id`, e.g. ``asst_85Ay7apSmArPOMKF4iRY893b``.  Set this value to the `config.llms.openai.asst_id` field to instruct `bibcat` to use this agent.

To tell bibcat to use the assistant, set `config.llms.openai.use_assistant=true`.

When submitting a paper to an assistant, the file or text is first uploaded to the Assistant.  The Assistant creates a temporary vector store database, reads the file content, and embeds it into the database.  The Assistant will search the file via the vector store when responding to your user prompt.

The Assistant supports uploading PDF or JSON files.  `bibcat` will accept either a local PDF file, as well as a bibcode entry from our source dataset.  When specifying a paper from the source dataset, using a bibcode or array index, `bibcat` will write the JSON data out to a temporary JSON file for upload.  The temporary filename syntax is of the form `temp_xxxx_(bibcode).json`, prefixed by `temp_` and suffixed by the bibcode of the paper.


## Response Output

The output response from the LLM prompt is written to a file, specified by `config.llms.prompt_output_file`, e.g. "paper_output.json".  The response output is organized by the name of the file, or the bibcode of the paper.  Repeated prompts using the same paper will be appended to the entry for that paper.

For example, running `bibcat run-gpt -b 2023Natur.616..266L` produces the following output:
```json
{
  "2023Natur.616..266L": [
    {
      "HST": [
        "MENTION",
        0.8
      ],
      "JWST": [
        "SCIENCE",
        0.95
      ]
    }
  ]
}
```

## Evaluating Output

To assess how well an LLM might be doing, we can try to evaulate it by running repeated trial runs, collecting the output, and comparing
to the human classifications from the source dataset.

First, run bibcat run-gpt with the `-n` flag to specify to run repeated submissions of the paper, and record all outputs in the output JSON file.

To submit paper index 2000, 10 times, run:
```bash
bibcat run-gpt -i 2000 -n 10
```
Once it's finished, you can evaluate the LLM output with:
```
bibcat evaluate-llm -i 2000
```

You should see some output similar to
```bash
Loading source dataset: /Users/bcherinka/Work/stsci/bibcat_data/dataset_combined_all_2018-2023.json
INFO - Evaluating output for 2022Sci...377.1211L
INFO - Number of runs: 10
INFO - 'Output Stats by LLM Mission and Paper Type:'
llm_mission llm_papertype  mean_llm_confidence  count  n_runs  accuracy  in_human_class
       JWST       MENTION                 0.50      2      10       0.0           False
       JWST       SCIENCE                 0.80      1      10       0.0           False
         K2       MENTION                 0.60      2      10       0.0           False
     KEPLER       MENTION                 0.55      4      10       0.0           False
       TESS       SCIENCE                 0.90     10      10     100.0            True
INFO - Missing missions by humans: JWST, KEPLER, K2
INFO - Missing missions by LLM:
```

For now this produces a Pandas dataframe, grouped by the LLM predicted mission and papertype, with its mean confidence score, the number of times that combination was output by the LLM, the total number of trial runs, an accuracy score of how well it matched the human classification, and a boolean flag if that combination appears in the human classification.  The human classication comes from the "class_missions" field in the source dataset file.






