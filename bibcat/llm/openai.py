import json
import os
import pathlib
import re

import openai
from openai import OpenAI
from openai.types.beta.assistant import Assistant

from bibcat import config
from bibcat.llm.io import get_source, get_file, get_llm_prompt
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)

class OpenAIHelper:
    """ Helper class for interacting with the OpenAI API

    Parameters
    ----------
    use_assistant : bool, optional
        Flag to use the OpenAI Assistant or not, by default None
    """

    def __init__(self, use_assistant: bool = None):
        """ init """
        self.use_assistant = use_assistant or config.llms.openai.use_assistant

        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.assistant = None
        self.vector_store = None
        self.file = None
        self.stores = []
        self.assistants = []
        self.original_response = None
        self.response = None

    def __repr__(self) -> str:
        return (f'<OpenAIHelper asst_id="{self.assistant.id if self.assistant else None}",'
                f'vs_id="{self.vector_store.id if self.vector_store else None}">')

    # upload the file
    def upload_file(self, file_path: str):
        """ Upload a file to the OpenAI API

        Parameters
        ----------
        file_path : str
            the path to a file

        Returns
        -------
        str
            the file id
        """
        self.file = self.client.files.create(
            file=open(file_path, "rb"),
            purpose='assistants'
        )

    def create_vector_store(self, name: str = 'Papers', files: list = None):
        """ Create a new vector store

        Parameters
        ----------
        name : str, optional
            the name of the vector store, by default 'Papers'
        files : list, optional
            a list of file_ids to attach to the store, by default None

        Returns
        -------
        VectorStore
            a new OpenAI vector store
        """
        self.vector_store = self.client.beta.vector_stores.create(
            name=name,
            file_ids=files
        )

    def get_vector_store(self, vs_id: str):
        """ Get a vector store by id

        Parameters
        ----------
        vs_id : str
            the id of the vector store

        Returns
        -------
        VectorStore
            the requested vector store
        """
        try:
            self.vector_store = self.client.beta.vector_stores.retrieve(vs_id)
        except openai.NotFoundError:
            pass

    def list_vector_stores(self) -> list[dict]:
        """ List all vector stores

        List all vector stores, and convert each response
        to a dictionary.

        Returns
        -------
        list[dict]
            a list of vector store dictionaries
        """
        for vs in self.client.beta.vector_stores.list():
            self.stores.append(vs.to_dict())
        return self.stores

    def create_assistant(self, name: str = 'Paper Reader', vs_id: str = None) -> Assistant:
        """ Create a new OpenAI assistant

        Creates a new OpenAI assistant with file search capabilities.  The llm model
        to use for the assistant is set in the config file by ``config.llms.openai.model``.
        Custom instructions and behavior for the assistant is set through a custom agent prompt file,
        or a config value, otherwise the default agent instructions will be used.
        See ``bibcat.llm.io.get_llm_prompt`` for more information.

        Parameters
        ----------
        name : str, optional, by default 'Paper Reader'
            the name of the assistant
        vs_id : str, optional
            the vector store id to attach, by default None

        Returns
        -------
        Assistant
            the new OpenAI assistant
        """
        # create a new vector store for the assistant, if none provided
        if not vs_id:
            self.create_vector_store()
            vs_id = self.vector_store.id

        # create the new assistant
        self.assistant = self.client.beta.assistants.create(
            name=name,
            instructions=get_llm_prompt('agent'),
            model=config.llms.openai.model,
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vs_id]}}
        )

    def list_assistants(self) -> list[dict]:
        """ List all assistants

        List all OpenAI Assistants, and convert each response
        to a dictionary.

        Returns
        -------
        list[dict]
            a list of assistant dictionaries
        """
        for aa in self.client.beta.assistants.list():
            self.assistants.append(aa.to_dict())

        return self.assistants

    def get_assistant(self, asst_id: str = None):
        """ Get an OpenAI Assistant

        Parameters
        ----------
        asst_id : str
            the assistant id

        Returns
        -------
        Assistant
            the requested assistant

        Raises
        ------
        ValueError
            when no assistant id is provided
        ValueError
            when the assistant for the given id is not found
        """
        asst_id = asst_id or config.llms.openai.asst_id
        if not asst_id:
            raise ValueError('No assistant id provided.  Either provide or set an assistant id, or first create a new assistant.')
        logger.info(f"Using assistant id: {asst_id}")

        try:
            self.assistant = self.client.beta.assistants.retrieve(asst_id)
        except openai.NotFoundError as e:
            raise ValueError(f"Assistant id {asst_id} not found.") from e

    def send_assistant_request(self, file_id: str, asst_id: str = None) -> dict:
        """ Send a prompt request to an OpenAI assistant

        Sends a user prompt request to an OpenAI assistant to search
        through a given file for content.  It retrieves the requested agent via
        the assistant id, ``asst_id``.  It creates a new message thread, attaching
        the input file id to the message thread, then submits the user prompt request.

        When attaching a file to a message thread, a temporary vector store is created, where the
        file is stored.  The assistant searches both the temporary vector store and any vector
        store attached to the assistant to answer the user prompt.

        The prompt response is then extracted and converted to JSON content. The response is
        stored in the instance ``response`` attribute.  The original message content
        can be found in the ``original_response`` attribute.

        At the end, the uploaded file and the temporary vector store are deleted.

        Parameters
        ----------
        file_id : str
            the file id of the uploaded file, to search on
        asst_id : str, optional
            the id of the assistant to use, by default None

        Returns
        -------
        dict
            the output respsonse from the assistant

        Raises
        ------
        ValueError
            when no assistant id is provided
        """

        # get an OpenAI Assistant
        self.get_assistant(asst_id)

        # create a new thread
        # attach the input file id to the message thread
        # this creates a temporary vector store for the file
        thread = self.client.beta.threads.create(
            messages=[
            {"role": "user", "content": get_llm_prompt('user'),
                "attachments": [
                { "file_id": file_id, "tools": [{"type": "file_search"}] }],
            }
            ]
        )

        # submit the prompt request
        logger.debug(f"File search thread: {thread.tool_resources.file_search}")
        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=self.assistant.id
        )

        # get the response content
        messages = list(self.client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
        message_content = messages[0].content[0].text
        logger.debug(f"Original response message content: {message_content.value}")
        self.original_response = message_content.value
        self.response = check_response(message_content.value)

        # do some cleanup; delete the file and the temporary vector store
        vs = thread.tool_resources.file_search.vector_store_ids[0]
        self.client.beta.vector_stores.delete(vs)
        self.client.files.delete(file_id)

        return self.response

    def populate_user_template(self, paper: dict) -> str:
        """ Format a user prompt template with paper data

        Parameters
        ----------
        paper : dict
            the input JSON paper content

        Returns
        -------
        str
            the fully formatted user prompt

        Raises
        ------
        ValueError
            when the prompt fields are missing from the paper data
        """
        user = get_llm_prompt('user')

        # check the user template fields match the paper dictionary keys
        fields = re.findall(r'{(.*?)}', user)
        missing = set(fields) - set(paper.keys())
        if missing:
            raise ValueError(f"Missing user template fields in input paper data: {missing}")

        # format the user prompt the paper content
        return user.format(**paper)

    def send_message(self, user_prompt: str = None) -> dict | str:
        """ Send a straight chat message to the LLM

        This
        Can pass a custom user prompt into method, otherwise it uses the prompt
        pulled from ``get_llm_prompt``.  The response is stored in the instance
        ``response`` attribute.  The original message content can be found in the
        ``original_response`` attribute.

        Parameters
        ----------
        user_prompt : str, optional
            A customized user prompt, by default None

        Returns
        -------
        dict | str
            the output respsonse from the model
        """
        result = self.client.chat.completions.create(
            model=config.llms.openai.model,
            messages=[{"role": "system", "content": user_prompt or get_llm_prompt('agent')},
                      {'role': 'user', 'content': get_llm_prompt('user')}])

        self.original_response = result.choices[0].message.content
        self.response = check_response(self.original_response)

        return self.response

    def submit_paper(self, file_path: str = None, bibcode: str = None, index: int = None) -> dict | str:
        """ Submit a paper to the OpenAI LLM model

        Submit a paper to the OpenAI LLM model for processing, either using an AI Assistant
        with file-search capability, or a straight chat message.

        Parameters
        ----------
        file_path : str, optional
            a path to a local file on disk, by default None
        bibcode : str, optional
            the bibcode of an entry in the source papetrack combined dataset, by default None
        index : int, optional
            a list item array index in the source papetrack combined dataset, by default None

        Returns
        -------
        dict | str
            The output response from the model for the given paper

        Raises
        ------
        ValueError
            when a file_path is given and the AI Assistant is not being used
        """

        if not self.use_assistant and file_path:
            raise ValueError("Cannot use a local file when not using the AI Assistant.")

        if self.use_assistant:
            # get the file path
            file_path = get_file(filepath=file_path, bibcode=bibcode, index=index)
            logger.info(f"Using file: {file_path}")

            # upload the file to openai
            self.upload_file(file_path)
            logger.info(f"Uploaded file id: {self.file.id}")

            # send the prompt request to the assistant
            response = self.send_assistant_request(self.file.id)
        else:
            paper = get_source(bibcode=bibcode, index=index)
            user_prompt = self.populate_user_template(paper)
            response = self.send_message(user_prompt=user_prompt)

        return response


def check_response(value: str) -> dict:
    """ Check the agent response

    Check the agent response for proper JSON content and
    extract.  If no JSON content is found, return an error message.

    Parameters
    ----------
    value : str
        the original agent response message_content.value

    Returns
    -------
    dict
        the extracted JSON content
    """
    if "```json" not in value:
        return value

    # extract the json content
    response = re.search(r'```json\n(.*?)\n```', value, re.DOTALL)

    if response:
        response = response.group(1)
        return json.loads(response)
    else:
        return {'error': 'No JSON content found in response'}


def write_output(filepath: str, response: dict):
    """ Write the output response to a file

    Writes the output json response to a file, located at
    $BIBCAT_OUTPUT/output/llms/openai_[config.llms.openai.model]/paper_output.json

    The output JSON file is organized by the filename or bibcode of the input file,
    with each prompt response appended in the relevant section.

    Parameters
    ----------
    filepath : str
        the name of the input file
    response : dict
        the response from the llm agent
    """

    # get the filename ; for sources, use the bibcode
    path = pathlib.Path(filepath)
    name = path.name
    if name.startswith('temp'):
        name = name.rsplit('_',1)[-1].split('.json')[0]

    # setup the output file
    out = pathlib.Path(config.paths.output) / f'llms/openai_{config.llms.openai.model}/{config.llms.prompt_output_file}'
    out.parent.mkdir(parents=True, exist_ok=True)

    # write the content
    if not os.path.exists(out):
        # create a new file
        data = {name: [response]}
        with open(out, 'w+') as f:
            json.dump(data, f, indent=2, sort_keys=False)
    else:
        # append to an existing file
        with open(out, 'r') as f:
            data = json.load(f)

        # append response to an existing file entry, or add a new one
        if name in data:
            data[name].append(response)
        else:
            data[name] = [response]

        # write the updated file
        with open(out, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=False)


def send_prompt(file_path: str = None, bibcode: str = None, index: int = None, n_runs: int = 1):
    """ Send a prompt to an OpenAI LLM model

    Parameters
    ----------
    file_path : str, optional
        a path to a local file on disk, by default None
    bibcode : str, optional
        the bibcode of an entry in the source papetrack combined dataset, by default None
    index : int, optional
        a list item array index in the source papetrack combined dataset, by default None
    n_runs : int, optional
        the number of runs to do, by default 1
    """
    oa = OpenAIHelper()

    # iterate for number of runs
    for i in range(n_runs):
        # get the file path
        file_path = get_file(filepath=file_path, bibcode=bibcode, index=index)
        logger.info(f"Using file: {file_path}")

        # upload the file to openai
        oa.upload_file(file_path)
        logger.info(f"Uploaded file id: {oa.file.id}")

        # send the prompt request to the assistant
        response = oa.send_assistant_request(oa.file.id)
        logger.info(f"Output: {response}")

        # write the output response to a file
        write_output(file_path, response)