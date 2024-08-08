
import json
import os
import pathlib
import tempfile

from bibcat import config
from bibcat.data.streamline_dataset import load_source_dataset
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)


def get_source(bibcode: str = None, index: int = None, body_only: bool = False) -> dict | str:
    """ Get the source dataset for a given bibcode or index.

    Retrieve the entry from the combined source dataset for a given bibcode or list index.

    Parameters
    ----------
    bibcode : str, optional
        the paper bibcode to retrieve, by default None
    index : int, optional
        the list item index to retrieve, by default None
    body_only : bool, optional
        Flag to only return the text body, by default False

    Returns
    -------
    dict | str
        a row from the source dataset
    """
    # load the source dataset
    source_dataset = load_source_dataset(do_verbose=False)
    n_sources = len(source_dataset)

    if bibcode:
        # get the source by bibcode
        res = [i for i in source_dataset if i['bibcode'] == bibcode]
        text = res[0] if res else None
        if not res:
            logger.warning('Requested bibcode not found in source datasets.')
    elif index is not None:
        index = int(index)
        # get the source by index
        text = source_dataset[index] if index < n_sources else None
        if index > n_sources:
            logger.warning('Requested index is out of range of the number of source datasets.')

    return text['body'] if text and body_only else text


def get_file(filepath: str = None, bibcode: str = None, index: int = None) -> str:
    """ Get a file path for paper data

    Get a file path of a paper to upload to an LLM.  If a file path is provided, e.g.
    a local pdf file, it is returned.  If a bibcode or index is provided, retrieves the
    source dataset and writes it out to a temporary json file.

    Parameters
    ----------
    filepath : str, optional
        a local filepath to a paper, by default None
    bibcode : str, optional
        the bibcode of a source paper, by default None
    index : int, optional
        the list index of a source paper, by default None

    Returns
    -------
    str
        the file path to the paper data
    """
    # if a real file, just return it
    if filepath and os.path.isfile(filepath):
        return filepath

    # if source dataset file, extract and create temporary file
    if bibcode or index is not None:
        source = get_source(bibcode=bibcode, index=index)
        bc = source['bibcode']

        # create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, prefix='temp_', suffix=f'_{bc}.json') as fp:
            fp.write(json.dumps(source, indent=2))
            fp.close()

        return fp.name


def get_llm_prompt(prompt_type: str) -> str:
    """ Get an LLM prompt

    Retrieve a user or agent prompt for an LLM from a file or the config. A user prompt
    is the text to be used as the input to the LLM, while the agent, or system, prompt is
    the text that defines the instructions or behavior of the LLM Agent to follow.  The agent
    prompt is only used when creating a new agent for the first time.

    You can define a custom user or agent prompt as a text file, located at
    $BIBCAT_DATA_DIR/llm_[prompt_type]_prompt.txt.  For example, place your custom user prompt
    at $BIBCAT_DATA_DIR/llm_user_prompt.txt. This file takes precendence. If no custom prompt file
    is found, the default user prompt will come from the config file field: ``llms.user_prompt``.

    To set an agent prompt, create a file at $BIBCAT_DATA_DIR/llm_agent_prompt.txt, and add your
    instructions for the agent.  If no custom agent prompt is found, a default agent prompt will
    be used. The default agent prompt will either come from the config file field: ``llms.agent_prompt``
    or from the default file at etc/default_agent_prompt.txt.

    Parameters
    ----------
    prompt_type : str
        The type of prompt to retrieve, either 'user' or 'agent'

    Returns
    -------
    str
        the text prompt

    Raises
    ------
    ValueError
        when an invalid prompt type is provided
    """

    if prompt_type not in {'user', 'agent'}:
        raise ValueError('Prompt type must be either "user" or "agent".')

    # if a prompt file exists, use it
    path = pathlib.Path(config.inputs[f'llm_{prompt_type}_prompt'])
    if path.exists():
        with open(path, 'r') as f:
            prompt = f.read()
            return prompt

    # otherwise, use the config user prompt and default agent prompt
    if prompt_type == 'user':
        prompt = config.llms.user_prompt
    elif prompt_type == 'agent':
        # see if there's a config agent prompt
        prompt = config.llms.agent_prompt

        # otherwise use the default
        if not prompt:
            default_agent = pathlib.Path(__file__).parent.parent / 'etc/default_agent_prompt.txt'
            with open(default_agent, 'r') as f:
                prompt = f.read()

    return prompt

