

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

You should see something like `INFO - Output: {'HST': ['MENTION', [0.3, 0.7]], 'JWST': ['SCIENCE', [0.9, 0.1]]}`. See [Response Output](#response-output) for details about the response output.  [Submitting a paper](#submitting-a-paper) describes the command for sending papers to OpenAI. For customizing and trialing new gpt prompts, see [User Configuration](#user-configuration) and [User and Agent (System) Prompts](#user-and-agent-system-prompts).


## Submitting a paper

The cli `bibcat run-gpt` submits a prompt with paper content.  The paper content can either be a local filepath on disk, or a bibcode or array list index from source dataset ``dataset_combined_all_2018-2023.json``.  When specifying a bibcode or list array index, the paper data is pulled from the source JSON dataset.

The `run-gpt` command submits a single paper. Note that when submitting any paper whose bibcode contains `&`, double quotes `" "` are needed for the bibcode in the command line.
```python
# submit a paper
bibcat run-gpt -f /Users/bcherinka/Downloads/2406.15083v1.pdf

# use a bibcode from the source dataset
bibcat run-gpt -b 2023Natur.616..266L
bibcat run-gpt -b "2022A&A...668A..63R"

# submit with entry 101 from the papertrack source dataset
bibcat run-gpt -i 101
```

The `run-gpt-batch` command submits a list of files.  The list can either be specified one of two ways:  1.) individually, with the `-f` option, as a filename, bibcode, or index, or 2.) a file that contains a list of filenames, bibcodes, or indices to use, with one line per entry.  By default, each paper in the list is submitted once.  You can instruct it to submit each paper multiple times with the `-n` option. Run `bibcat run-gpt-batch --help` to see the full list of cli options.

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
2024-08-22 15:49:56,781 - bibcat.llm.openai - INFO - Output: {'HST': ['MENTION', [0.3, 0.7]], 'JWST': ['SCIENCE', [0.9, 0.1]]}
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

>```txt
>You are an professional expert on whales.
>You are also witty and always respond with a clever pun.
>*Always* format your response as valid JSON, with the following example as a guide:
>
>```json
>{{
> "whale": "Blue",
> "response": "The Blue whale weighs up to 199 tons.",
> "source": "https://en.wikipedia.org/wiki/Blue_whale"
>}}```
>
> *Always* include a "source" field, which is real url link to your source of information
>

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

The output response from the LLM prompt is written to a file, specified by `config.llms.prompt_output_file`, e.g. "paper_output.json".
The response output is organized by the name of the file, or the bibcode of the paper.  Repeated prompts using the same
paper will be appended to the entry for that paper.

By default bibcat uses [Structured Response](https://openai.com/index/introducing-structured-outputs-in-the-api/), defining a Pydantic response model as the `response_format`.  The structure of the response is organized as follows:

- notes: Notes on the LLM's thought process and decision making
- missions: a list of mission-papertype classifications
  - mission: the name of the mission class
  - papertype: the type of paper classification
  - confidence: an array of the LLM confidence values of ["science", "mention"]
  - reason: the LLMs rationale for why it's assigning the mission-papertype
  - quotes: if able, a list of direct quotes from the paper that back up the LLM's reason.  (These quotes may be hallucinated!)

For example, running `bibcat run-gpt -b "2023Natur.616..266L"` produces the following output:
```json
  "2023Natur.616..266L": [
    {
      "notes": "I reviewed the paper and found multiple references to both JWST and HST. The JWST is explicitly noted for
      the new observations, while HST is referenced in the context of overlapping imaging with JWST's observations.
      MAST data are explicitly mentioned as part of the data processing steps.",
      "missions": [
        {
          "mission": "JWST",
          "papertype": "SCIENCE",
          "confidence": [
            0.95,
            0.05
          ],
          "reason": "The paper presents and analyzes new observational data from JWST's CEERS program.",
          "quotes": [
            "This article is based on the first imaging taken with the NIRCam on JWST as part of the CEERS
            program (Principal Investigator, Finkelstein; Program Identifier, 1345)."
          ]
        },
        {
          "mission": "HST",
          "papertype": "MENTION",
          "confidence": [
            0.1,
            0.9
          ],
          "reason": "HST data are referenced primarily for comparative purposes and indicate overlap with JWST
          observations, but no new HST data is presented.",
          "quotes": [
            "The total area covered by these initial data is roughly 40 arcmin 2 and overlaps fully with the
            existing HST\u2013Advanced Camera for Surveys (ACS) and WFC3 footprint."
          ]
        }
      ]
    },
```
You can turn off structured response output with the `-u` flag, e.g. `bibcat run-gpt -b "2023Natur.616..266L" -u`.

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
bibcat evaluate-llm -b "2020A&A...642A.105K"
```

You should see some output similar to
```bash
Loading source dataset: /Users/jyoon/Documents/asb/bibliography_automation/bibcat_datasets//dataset_combined_all_2018-2023.json
INFO - Evaluating output for 2020A&A...642A.105K
INFO - Number of runs: 3
INFO - Human Classifications:
 KEPLER: SCIENCE
Output Stats by LLM Mission and Paper Type:
llm_mission llm_papertype mean_llm_confidences std_llm_confidences  count  n_runs  weighted_confs  consistency  in_human_class  mission_in_text  hallucination_by_llm
         K2       MENTION           [0.2, 0.8]          [0.0, 0.0]      1       3  [0.067, 0.267]          0.0          False            False                  True
         K2       SCIENCE         [0.85, 0.15]        [0.05, 0.05]      2       3  [0.567, 0.100]          0.0          False            False                  True
     KEPLER       SCIENCE         [0.92, 0.08]        [0.02, 0.02]      3       3  [0.920, 0.080]        100.0          True              True                 False
INFO - Missing missions by humans: K2
INFO - Missing missions by LLM:
INFO - Hallucination by LLM: K2
Writing output to /Users/jyoon/GitHub/bibcat/output/output/llms/openai_gpt-4o-mini/summary_output_t0.7.json
```

The output is also written to a file specified by `config.llms.eval_output_file`.

For now this produces a Pandas dataframe, grouped by the LLM predicted mission and papertype, with its mean confidence score, the number of times that combination was output by the LLM, the total number of trial runs, frequency-weighted confidence values, an accuracy score of how well it matched the human classification, and a boolean flag if that combination appears in the human classification.  The human classication comes from the "class_missions" field in the source dataset file.

Alternatively, you can both submit a paper for classfication and evaluate it in a single command using the `-s`, `--submit` flag.  In combination with the `-n` flag,
this will classify the paper `num_runs` time before evaluation.

This example first classifies paper index 1000, 20 times, then evaluates the output.
```base
bibcat evaluate-llm -i 1000 -s -n 20
```

```bash
Loading source dataset: /Users/bcherinka/Work/stsci/bibcat_data/dataset_combined_all_2018-2023.json
INFO - Using paper bibcode: 2022SPIE12184E..24M
INFO - Output: {'JWST': ['MENTION', 0.7]}
INFO - Using paper bibcode: 2022SPIE12184E..24M
INFO - Output: {'JWST': ['MENTION', 0.5]}
INFO - Using paper bibcode: 2022SPIE12184E..24M
INFO - Output: {'JWST': ['MENTION', 0.7]}
...
INFO - Evaluating output for 2022SPIE12184E..24M
INFO - Number of runs: 20
INFO - Human Classifications:
 JWST: MENTION
INFO - Output Stats by LLM Mission and Paper Type:
llm_mission llm_papertype  mean_llm_confidence  std_llm_confidence  count  n_runs  consistency  in_human_class   mission_in_text
       JWST       MENTION                  0.5            0.107606     20      20        100.0            True            True
INFO - Missing missions by humans:
INFO - Missing missions by LLM:
```

### Output Columns

Definitions of the output columns from the evaluation.
#### Summary output
- **human**: Human classifications
- **threshold_acceptance**: The threshold value to accept the llm's classifications
- **threshold_inspection**: The threshold value to require human inspection
- **llm**: llm's classification whose confidence value is higher than or equal to the threshold value. Each entry is organized as:
  - "mission": "papertype" (the mission and papertype classification)
  - **confidence**: the list of final LLM confidences values for [science, mention] papertype classification
  - **probability**: the probability that the specified mission is relevant to the paper
- **inspection**: The list of missions/papertypes for human inspection due to the edge-case confidence values (e.g, 0.5)
- **missing_by_human**: The set of missing missions by human classification
- **missing_by_llm**: The set of missing missions by llm classification
- **hallucinated_missions**: The list of missions hallucinated by llm

#### Each mission/papertype DataFrame output, as "df"
- **llm_mission**: The mission from the LLM output
- **mean_llm_confidence**: The list of the mean confidence values of SCIENCE and MENTION across all trial runs, for each mission + papertype combination. Conditional probabilities. Sum to 1.
- **std_llm_confidence**: The standard deviation of the confidence values of SCIENCE and MENTION  across all trial runs
- **count**: The number of times a mission + papertype combo was included in the LLM response, across all trial runs
- **llm_papertype**: The papertype from the LLM output
- **n_runs**: The total number of trial runs
- **weighted_confs**: Frequency-weighted confidence values.  The "mean_llm_confidence" scaled by the fraction of runs in which the mission+papertype appeared. Combined measure of frequency and confidence.
- **consistency**: The percentage of how often the LLM mission + papertype matched the human classification
- **in_human_class**: Flag whether or not the mission + papertype was included in the set of human classifications
- **mission_in_text**: Flag whether or not the mission keyword is in the source paper text
- **hallucination_by_llm**: Flag whether or not the mission keyword is hallucinated by LLM

#### Output Statistics by each mission, as "mission_conf"
- **llm_mission**: The mission from the LLM output
- **total_mission_conf**: The total confidence value for the given mission.  Sum of all weighted [science, mention] conf values.
- **total_weighted_conf**: The total weighted confidence values for the given mission, by [science, mention]
- **prob_mission**: The probability the input mission is relevant to the paper, i.e. "overall mission confidence"
- **prop_papertype**: Within each mission, the probability the mission is a science vs mention papertype

### Example Output

An example output file would look like:
```json
{
  "2022Sci...377.1211L": {
    "human": {
      "TESS": "SCIENCE"
    },
    "threshold_acceptance": 0.7,
    "threshold_inspection": 0.5,
    "llm": [
      {
        "TESS": "SCIENCE",
        "confidence": [
          0.809,
          0.191
        ],
        "probability": 0.98
      }
    ],
    "inspection": [],
    "missing_by_human": [
      "HST"
    ],
    "missing_by_llm": [],
    "hallucinated_missions": [
      "HST"
    ],
    "df": [
      {
        "llm_mission": "HST",
        "llm_papertype": "SCIENCE",
        "mean_llm_confidences": [
          0.8,
          0.2
        ],
        "std_llm_confidences": [
          0.0,
          0.0
        ],
        "count": 1,
        "n_runs": 50,
        "weighted_confs": [
          0.016,
          0.004
        ],
        "consistency": 0.0,
        "in_human_class": false,
        "mission_in_text": false,
        "hallucination_by_llm": true
      },
      {
        "llm_mission": "TESS",
        "llm_papertype": "MENTION",
        "mean_llm_confidences": [
          0.23,
          0.78
        ],
        "std_llm_confidences": [
          0.11,
          0.11
        ],
        "count": 4,
        "n_runs": 50,
        "weighted_confs": [
          0.018,
          0.062
        ],
        "consistency": 0.0,
        "in_human_class": false,
        "mission_in_text": true,
        "hallucination_by_llm": false
      },
      {
        "llm_mission": "TESS",
        "llm_papertype": "SCIENCE",
        "mean_llm_confidences": [
          0.86,
          0.14
        ],
        "std_llm_confidences": [
          0.05,
          0.05
        ],
        "count": 46,
        "n_runs": 50,
        "weighted_confs": [
          0.791,
          0.129
        ],
        "consistency": 92.0,
        "in_human_class": true,
        "mission_in_text": true,
        "hallucination_by_llm": false
      }
    ],
    "mission_conf": [
      {
        "llm_mission": "HST",
        "total_mission_conf": 0.02,
        "total_weighted_conf": [
          0.016,
          0.004
        ],
        "prob_mission": 0.02,
        "prob_papertype": [
          0.8,
          0.2
        ]
      },
      {
        "llm_mission": "TESS",
        "total_mission_conf": 1.0,
        "total_weighted_conf": [
          0.809,
          0.191
        ],
        "prob_mission": 0.98,
        "prob_papertype": [
          0.809,
          0.191
        ]
      }
    ]
  }
}
```


### Batch Evaluation

You can batch evaluate a list of papers/bibcodes with the `evaluate-llm-batch` command.  For example, to submit a list of bibcodes to `run-gpt-batch`,
with 20 runs each paper, then batch evaluate them, run:
```bash
bibcat evaluate-llm-batch -p bibcode_list.txt -s -n 20
```

## Plotting Evaluation Plots

You can assess model performance using confusion matrix plots or Receiver Operating Characteristic (ROC) curves. A [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) plot helps evaluation classification model performance by showing true positives ($TP$), true negatives ($TN$), false positives ($FP$), and false negatives ($FN$), making it useful for understanding evaluation metrics such as accuracy ($\frac{TP+TN}{TP+TB+FP+FN}$), precision ($\frac{TP}{TP+FP}$), recall (sensitivity, $\frac{TP}{(TP+FN)}$), and F1 score ($2\times \frac{precision \times recall}{ precison + recall}$). A [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) curve evaluates a model's ability to distinguish between classes by plotting the true positive rate against the false positive rate at various thresholds, with the area under the curve (AUC) which represents the degree of separability between classes. For instance, AUC=1.0 indicates perfect and AUC =0.5 is as good as random guessing. To provide more reliable and stable performance metrics, larger datasets (hundreds or thousands) are recommended. With small datasets, you make interpreations less reliable.

### Confusion Matrix Plot
To plot a confusion matrix for specific missions, run:
```bash
bibcat eval-plot -c -m HST -m JWST
```
To plot a confusion matrix for all missions, run:
```bash
bibcat eval-plot -c -a
```
### Receiver Operating Characteristic (ROC) Plot
To plot a confusion matrix for specific missions, run:
```bash
bibcat eval-plot -r -m HST -m JWST
```
To plot a confusion matrix for all missions, run:
```bash
bibcat eval-plot -r -a
```

## Statistics output
After running bibcat GPT classification using bibcat run-gpt, bibcat run-gpt-batch, or bibcat evaluate-llm to generate classifications for papers, you may want to review various statistics. These statistics can include the number of papers per mission and papertype, the count of accepted papertypes, and the number of papers that require human inspection due to low confidence scores.

From the evaluation summary output file, you may want to see the list of the papers where human classifications are not consistent with llm classifications. The next command line will also create this file.

The filenames are defined in `bibcat_config.yaml`: `eval_output_file`, `ops_output_file`, and `inconsistent_classification_file`.
To create a statistics JSON file, use the command line options listed below.

### Evaluation summary statistics for mission+papertype pairs

To create a statisitics output from the *e*valuation summary output, e.g., `summary_output_t0.7.json`, run:
```bash
bibcat stats-llm -e
```
The output file name will be something like `evaluation_stats_t0.7.json` where `t0.7` refers to the threshold value, `0.7` to accept the llm's papertype.

This command line will also create a file for the list of the papers where human classifications are not consistent with llm classifications.

### Operation classification statistics for mission+papertype pairs
This output will provide various number counts including the number of papers with accepted papertype which meets this condition `threshold_acceptance >= confidence` and the number of papers required for human inspection (`threshold_inspection <= confidence < threshold_acceptance`) for final papertype assignment. It also includes the lists of bibcodes of accepted papertypes and inspection required for human inspection.

To create a statisitics output,`operation_stats_t0.7.json`  from the llm classification output for *o*peration, `paper_output.json`, run:
```bash
bibcat stats-llm -o
```
```bash
INFO - reading /Users/jyoon/GitHub/bibcat/output/output/llms/openai_gpt-4o-mini/paper_output.json
INFO - threshold for accepting llm classification: 0.7
INFO - threshold for inspecting llm classification: 0.4
INFO - Production counts by LLM Mission and Paper Type:
   mission papertype  total_count  accepted_count  inspection_count
     GALEX   MENTION            4               4                 0
     GALEX   SCIENCE            2               2                 0
       HST   MENTION           22              21                 1
       HST   SCIENCE            4               4                 0
      JWST   MENTION            8               8                 0
      JWST   SCIENCE           17              17                 0

```
### Statistics Output columns
Both the evaluation and operation statistics files share the same column names.


#### Statistics Output
The definitions of the output columns are following.

- **threshold_acceptance**: The threshold value to accept the LLM papertype classification
- **threshold_inspection**: The threshold value to require human inspection
- **mission**: MAST mission
- **papertype**: papertype classified by LLM
- **total_count**: The total number of papers
- **accepted_count**: The count of papers with accepted llm papertype
- **accepted_bibcodes**: The bibcode list of the papers with accepted llm papertype
- **inspection_count**: The count of papers required for human inspection
- **inspection_bibcodes**: The bibcode list of the papers for papertype required human inspection

#### File output for the inconsistent classifications
The output columns are `mission`, `papertype`, `mean_llm_confidences`,`bibcode`, `in_human_class`, `mission_in_text`, and `consistency`.
The column definitions can be found in the [Output Columns](#output-columns).
