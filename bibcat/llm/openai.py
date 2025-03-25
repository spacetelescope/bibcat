import json
import os
import pathlib
import re

import openai
from openai import OpenAI
from openai.types.beta.assistant import Assistant
from pydantic import BaseModel, Field

from bibcat import config
from bibcat.llm.evaluate import identify_missions_in_text
from bibcat.llm.io import get_file, get_llm_prompt, get_source, write_output
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)


class MissionInfo(BaseModel):
    """Pydantic model for a mission entry"""

    mission: str = Field(..., description="The name of the mission.")
    papertype: str = Field(..., description="The type of paper you think it is")
    confidence: list[float] = Field(..., description="A list of float values of your confidence")
    reason: str = Field(
        ..., description="A short sentence summarizing your reasoning for classifying this mission + papertype"
    )
    quotes: list[str] = Field(..., description="A list of exact quotes from the paper that support your reason")


class InfoModel(BaseModel):
    """Pydantic model for the parsed response from the LLM"""

    notes: str = Field(..., description="all your notes and thoughts you have written down during your process")
    missions: list[MissionInfo] = Field(..., description="a list of your identified missions")


class OpenAIHelper:
    """Helper class for interacting with the OpenAI API

    Parameters
    ----------
    use_assistant : bool, optional
        Flag to use the file-search OpenAI Assistant or not, by default None
    verbose : bool, optional
        Flag to turn on verbose logging, by default None
    structured : bool, optional
        Flag to use structured response, by default True
    """

    def __init__(self, use_assistant: bool = None, verbose: bool = None, structured: bool = True):
        """init"""
        # input parameters
        self.use_assistant = use_assistant or config.llms.openai.use_assistant
        self.verbose = verbose or config.logging.verbose
        self.structured = structured

        # llm attributes
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.assistant = None
        self.vector_store = None
        self.file = None
        self.stores = []
        self.assistants = []
        self.original_response = None
        self.response = None
        self.response_classes = None
        self.user_prompt = None
        self.agent_prompt = None

        # paper attributes
        self.filename = None
        self.bibcode = None
        self.paper = None

    def __repr__(self) -> str:
        return (
            f'<OpenAIHelper use_assistant="{self.use_assistant}",'
            f'asst_id="{self.assistant.id if self.assistant else None}",'
            f'vs_id="{self.vector_store.id if self.vector_store else None}">'
        )

    # upload the file
    def upload_file(self, file_path: str):
        """Upload a file to the OpenAI API

        Parameters
        ----------
        file_path : str
            the path to a file

        Returns
        -------
        str
            the file id
        """
        self.file = self.client.files.create(file=open(file_path, "rb"), purpose="assistants")

    def create_vector_store(self, name: str = "Papers", files: list = None):
        """Create a new vector store

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
        self.vector_store = self.client.beta.vector_stores.create(name=name, file_ids=files)

    def get_vector_store(self, vs_id: str):
        """Get a vector store by id

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
        """List all vector stores

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

    def create_assistant(self, name: str = "Paper Reader", vs_id: str = None) -> Assistant:
        """Create a new OpenAI assistant

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
            instructions=get_llm_prompt("agent"),
            model=config.llms.openai.model,
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vs_id]}},
        )

    def list_assistants(self) -> list[dict]:
        """List all assistants

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
        """Get an OpenAI Assistant

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
            raise ValueError(
                "No assistant id provided.  Either provide or set an assistant id, or first create a new assistant."
            )
        logger.info(f"Using assistant id: {asst_id}")

        try:
            self.assistant = self.client.beta.assistants.retrieve(asst_id)
        except openai.NotFoundError as e:
            raise ValueError(f"Assistant id {asst_id} not found.") from e

    def send_assistant_request(self, file_id: str, asst_id: str = None) -> dict:
        """Send a prompt request to an OpenAI assistant

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
                {
                    "role": "user",
                    "content": get_llm_prompt("user"),
                    "attachments": [{"file_id": file_id, "tools": [{"type": "file_search"}]}],
                }
            ]
        )

        # submit the prompt request
        logger.debug(f"File search thread: {thread.tool_resources.file_search}")
        run = self.client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id=self.assistant.id)

        # get the response content
        messages = list(self.client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
        message_content = messages[0].content[0].text
        logger.debug(f"Original response message content: {message_content.value}")
        self.original_response = message_content.value
        self.response = extract_response(message_content.value)

        # do some cleanup; delete the file and the temporary vector store
        vs = thread.tool_resources.file_search.vector_store_ids[0]
        self.client.beta.vector_stores.delete(vs)
        self.client.files.delete(file_id)

        return self.response

    def populate_user_template(self, paper: dict) -> str:
        """Format a user prompt template with paper data

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
        user = get_llm_prompt("user")

        # check the user template fields match the paper dictionary keys
        fields = re.findall(r"{(.*?)}", user)
        missing = set(fields) - (set(paper.keys()) | set(["missions"]))
        if missing:
            logger.warning("Missing user template fields in input paper data: %s. Filling empty values.", missing)
            paper.update(dict.fromkeys(missing, ""))

        # get the text keyword match for missions
        mm = self.get_mission_text(paper)
        missions = ", ".join(mm.keys())

        # format the user prompt the paper content
        return user.format(**paper, missions=missions)

    def get_mission_text(self, paper) -> dict:
        """Get flags for missions found in paper text"""
        # fmt: off
        missions = ["HST", "JWST", "Roman", "HLA", "HSC", "TESS", "KEPLER", "K2", "GALEX", "PanSTARRS",
                    "FUSE", "IUE", "HUT", "UIT", "WUPPE", "BEFS", "TUES", "IMAPS", "EUVE"]
        return {k: v for k, v in zip(missions, identify_missions_in_text(missions, text=paper['body'])) if v}

    def send_message(self, user_prompt: str = None) -> dict | str:
        """Send a straight chat message to the LLM

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
            messages=[
                {"role": "system", "content": get_llm_prompt("agent")},
                {"role": "user", "content": user_prompt or get_llm_prompt("user")},
            ],
        )

        self.original_response = result.choices[0].message.content
        self.response = extract_response(self.original_response)

        return self.response

    def send_structured_message(self, user_prompt: str = None) -> dict | str:
        """Send a chat message to the LLM using Structured Response

        Sends your prompt to the LLM model with an expected response format
        of InfoModel.  The LLM will parse its response into the structure you
        provide. See https://openai.com/index/introducing-structured-outputs-in-the-api/
        Works with minimum gpt-4o-mini-2024-07-18 and gpt-4o-2024-08-06 models, but
        structured outputs with response formats is available on gpt-4o-mini and gpt-4o-2024-08-06 and
        any fine tunes based on these models.

        Parameters
        ----------
        user_prompt : str, optional
            A customized user prompt, by default None

        Returns
        -------
        dict | str
            the output response from the model

        Raises
        ------
        ValueError
            when the model is not one of the supported models
        """

        result = self.client.beta.chat.completions.parse(
            model=config.llms.openai.model,
            messages=[
                {"role": "system", "content": get_llm_prompt("agent")},
                {"role": "user", "content": user_prompt or get_llm_prompt("user")},
            ],
            response_format=InfoModel,
        )

        self.original_response = result.choices[0].message.content

        message = result.choices[0].message
        if message.parsed:
            self.response = message.parsed.model_dump()
        else:
            self.response = message.refusal

        return self.response

    def submit_paper(self, filepath: str = None, bibcode: str = None, index: int = None) -> dict | str:
        """Submit a paper to the OpenAI LLM model

        Submit a paper to the OpenAI LLM model for processing, either using an AI Assistant
        with file-search capability, or a straight chat message.

        Parameters
        ----------
        filepath : str, optional
            a path to a local input paper file on disk, by default None
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

        if not self.use_assistant and filepath:
            raise ValueError("Cannot use a local file when not using the AI Assistant.")

        # set the user / agent prompts
        self.user_prompt = get_llm_prompt("user")
        self.agent_prompt = get_llm_prompt("agent")

        if self.use_assistant:
            # get the file path
            self.filename = get_file(filepath=filepath, bibcode=bibcode, index=index)
            logger.info(f"Using file: {self.filename}")

            # upload the file to openai
            self.upload_file(self.filename)
            logger.info(f"Uploaded file id: {self.file.id}")

            # send the prompt request to the assistant
            response = self.send_assistant_request(self.file.id)
        else:
            # get the paper source
            self.paper = get_source(bibcode=bibcode, index=index)
            if not self.paper:
                self.bibcode = bibcode
                logger.warning(f"No paper source found for bibcode: {bibcode}")
                return {"error": f"Bibcode {bibcode} not found in source data."}

            self.bibcode = bibcode or self.paper.get("bibcode")
            logger.info(f"Using paper bibcode: {self.bibcode}")

            # populate the user template with paper data
            self.user_prompt = self.populate_user_template(self.paper)

            # send the prompt
            if self.structured or config.llms.openai.model in ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06"]:
                # automatically use the structured response if we're using the right models
                logger.info("Using structured response.")
                response = self.send_structured_message(user_prompt=self.user_prompt)
            else:
                # otherwise, use the regular response
                logger.info("Using unstructured response.")
                response = self.send_message(user_prompt=self.user_prompt)

        return response

    def get_output_key(self):
        """Get the output key for writing the response to a file

        Returns either the name of a file or the bibcode of a paper source.
        This key is used to organize the output JSON file content.

        """
        if self.bibcode:
            return self.bibcode

        if self.filename:
            path = pathlib.Path(self.filename)
            name = path.name
            # extract bibcode from temp file
            if name.startswith("temp"):
                name = name.rsplit("_", 1)[-1].split(".json")[0]

            return name


def extract_response(value: str) -> dict:
    """Extract the agent response

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
    # extract the json content
    response = re.search(r"```json\n(.*?)\n```", value, re.DOTALL)

    if response:
        response = response.group(1)
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            return {"error": f'Error decoding JSON content: "{e}"'}
    else:
        return {"error": "No JSON content found in response"}


def convert_to_classification(output: dict, bibcode: str, threshold: float = 0.5) -> dict:
    """Convert response to classification format

    Converts the JSON response to conform to the format from the
    data source ``class_missions`` field.

    Parameters
    ----------
    output : dict
        the JSON response
    bibcode : str
        the bibcode of the paper
    threshold : float, optional
        the threshold for rejection, by default 0.5

    Returns
    -------
    dict
        the formatted classification output
    """
    if "error" in output:
        logger.warning("Error in prompt JSON response. Cannot convert output.")
        return None

    try:  # NEED to REVISIT what to do when max(p[0], p[1]) == 0.5
        class_missions = {
            k: {"bibcode": bibcode, "papertype": papertype}
            for k, [papertype, p] in output.items()
            if max(p[0], p[1]) >= threshold or max(p[0], p[1]) == 0.5
        }
    except ValueError as e:
        logger.warning(f"Error converting output to classification format: {e}")
        return None
    else:
        return class_missions


def classify_paper(
    file_path: str = None,
    bibcode: str = None,
    index: int = None,
    n_runs: int = 1,
    use_assistant: bool | None = None,
    verbose: bool = None,
    structured: bool = True,
):
    """Send a prompt to an OpenAI LLM model to classify a paper

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
    use_assistant : bool, optional
        Flag to use the OpenAI file-search Assistant or not, by default None
    verbose : bool, optional
        Flag to turn on verbose logging, by default None
    structured : bool, optional
        Flag to use structured response, by default True
    """
    oa = OpenAIHelper(use_assistant=use_assistant, verbose=verbose, structured=structured)

    # iterate for number of runs
    for i in range(n_runs):
        # submit the paper to the LLM
        response = oa.submit_paper(filepath=file_path, bibcode=bibcode, index=index)

        # log the prompts if verbosity set
        if oa.verbose:
            logger.info(f"Agent Prompt: {oa.agent_prompt}")
            logger.info(f"User Prompt: {oa.user_prompt}")
            logger.info(f"Original Prompt Response: {oa.original_response}")

        logger.info(f"Output: {response}")

        # write the output response to a file
        key = oa.get_output_key()
        write_output(key, response)
