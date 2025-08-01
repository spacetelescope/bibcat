import json
import os
import pathlib
import re
from enum import Enum

import openai
from openai import OpenAI
from openai.lib._parsing._responses import type_to_text_format_param
from pydantic import BaseModel, Field, ValidationError, field_serializer, field_validator

from bibcat import config
from bibcat.llm.evaluate import identify_missions_in_text
from bibcat.llm.io import get_file, get_llm_prompt, get_source, write_output
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)

# create an enum for the missions, using the config list
MissionEnum = Enum("MissionEnum", dict(zip(map(str.lower, config.missions), config.missions)))


class PapertypeEnum(str, Enum):
    """Enumeration of paper types for classification"""

    science = "SCIENCE"
    mention = "MENTION"


class MissionInfo(BaseModel):
    """Pydantic model for a mission entry"""

    mission: MissionEnum = Field(..., description="The name of the mission.")
    papertype: PapertypeEnum = Field(..., description="The type of paper you think it is")
    quotes: list[str] = Field(..., description="A list of exact quotes from the paper that support your reason")
    reason: str = Field(
        ..., description="A short sentence summarizing your reasoning for classifying this mission + papertype"
    )
    confidence: list[float] = Field(
        ..., description="Two float values representing confidence for SCIENCE and MENTION. Must sum to 1.0."
    )

    @field_serializer("mission", "papertype")
    def serialize_enums(self, item: Enum):
        """Serialize the enums to their value"""
        return item.value

    @field_validator("confidence", mode="after")
    @classmethod
    def validate_confidence(cls, value: list[float]) -> list[float]:
        """Ensure the confidence is a list of two floats that sum to 1"""
        if len(value) != 2:
            raise ValueError("Confidence must contain exactly two float values.")
        if abs(sum(value) - 1.0) > 1e-6:
            raise ValueError(f"Confidence values must sum to 1.0, got {sum(value):.6f}.")
        return value

    @field_serializer("quotes")
    def strip_quotes(self, value: list[str]) -> list[str]:
        """Strip lead/trail quotes from the quotes list"""
        return [quote.strip('"') for quote in value]


class InfoModel(BaseModel):
    """Pydantic model for the parsed response from the LLM"""

    notes: str = Field(..., description="all your notes and thoughts you have written down during your process")
    missions: list[MissionInfo] = Field(..., description="a list of your identified missions")


class OpenAIHelper:
    """Helper class for interacting with the OpenAI API

    Parameters
    ----------
    verbose : bool, optional
        Flag to turn on verbose logging, by default None
    """

    def __init__(self, verbose: bool = None):
        """init"""
        # input parameters
        self.verbose = verbose or config.logging.verbose

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
        self.batch = None

        # paper attributes
        self.filename = None
        self.bibcode = None
        self.paper = None

    def __repr__(self) -> str:
        return f'<OpenAIHelper vs_id="{self.vector_store.id if self.vector_store else None}">'

    # upload the file
    def upload_file(self, file_path: str, purpose: str = "assistants"):
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
        self.file = self.client.files.create(file=open(file_path, "rb"), purpose=purpose)

    def retrieve_file(self, filename: str):
        """Retrieve a file from the OpenAI API

        Retrieves a file from OpenAI matching on the filename. If no file is found,
        it uploads the file instead.  The input filename can either be a full path
        or just the filename.

        Parameters
        ----------
        filename : str
            the filename to get
        """
        files = [i for i in self.client.files.list() if i.filename in filename]
        if files:
            self.file = files[0]
        else:
            self.upload_file(filename)

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

        missing = set(fields) - (set(paper.keys()) | set(["missions", "kw_missions"]))
        if missing:
            logger.warning("Missing user template fields in input paper data: %s. Filling empty values.", missing)
            paper.update(dict.fromkeys(missing, ""))

        # get the text keyword match for missions
        mm = self.get_mission_text(paper)
        kw_missions = ", ".join(mm.keys())

        # format the user prompt the paper content
        return user.format(**paper, missions=", ".join(config.missions), kw_missions=kw_missions)

    def get_mission_text(self, paper: dict) -> dict:
        """Get flags for missions found in paper text"""

        text = f"{paper['title'][0]}; {paper.get('abstract', '')}; {paper['body']}"

        return {k: v for k, v in zip(config.missions, identify_missions_in_text(config.missions, text=text)) if v}

    def send_message(self, user_prompt: str = None, with_file: bool = None):
        """Send a chat message to the LLM

        Sends your prompt to the LLM model with an expected structured response format
        of InfoModel.  The LLM will parse its response into the structure you
        provide. See https://openai.com/index/introducing-structured-outputs-in-the-api/
        Works with minimum gpt-4o-mini-2024-07-18 but all newer models of gpt-4o and all
        o-models support structured output.

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
        # set the proper llm input
        llm_input = user_prompt or get_llm_prompt("user")
        if with_file:
            llm_input = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file_id": self.file.id,
                        },
                        {
                            "type": "input_text",
                            "text": llm_input,
                        },
                    ],
                }
            ]

        # send the request
        try:
            result = self.client.responses.parse(
                model=config.llms.openai.model,
                instructions=get_llm_prompt("agent"),
                input=llm_input,
                text_format=InfoModel,
            )
        except openai.BadRequestError as e:
            logger.error("Error sending request: %s", e)
            return {"error": f"Error sending request: {e}"}

        except ValidationError as ve:
            logger.error("Pydantic Validation error parsing response: %s", ve)
            return {"error": f"Pydantic Validation error: {ve}"}

        if result.error:
            self.response = result.error
        else:
            self.response = result.output_parsed.model_dump()
        return self.response

    def submit_paper(
        self, filepath: str = None, bibcode: str = None, index: int = None, paper_dict: dict = None
    ) -> dict | str:
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
        paper_dict : dict, optional
            paper dictionary, for example,
            paper_dict = {"bibcode": "bibcode", "title": ["title",], "abstract": "abstract", "body": "body"}

        Returns
        -------
        dict | str
            The output response from the model for the given paper

        Raises
        ------
        ValueError
            when a file_path is given and the AI Assistant is not being used
        """

        # set the user / agent prompts
        self.user_prompt = get_llm_prompt("user")
        self.agent_prompt = get_llm_prompt("agent")

        # check for a file path
        with_file = filepath is not None
        # check for a paper text dictionary from the PaperTrack pipeline
        with_paper_dict = paper_dict is not None

        if with_file:
            # get the file path
            self.filename = get_file(filepath=filepath, bibcode=bibcode, index=index)
            logger.info("Using file: %s", self.filename)

            # upload the file to openai
            self.upload_file(self.filename)
            logger.info("Uploaded file id: %s", self.file.id)

        elif with_paper_dict:
            # get the paper text dictionary from the PaperTrack pipeline
            self.bibcode = paper_dict["bibcode"]
            self.paper = paper_dict
            logger.info("Using paper_dict in PaperTrack: %s", self.bibcode)

            # populate the user template with text
            self.user_prompt = self.populate_user_template(self.paper)
        else:
            # get the paper source
            self.paper = get_source(bibcode=bibcode, index=index)
            if not self.paper:
                self.bibcode = bibcode
                logger.warning("No paper source found for bibcode: %s", bibcode)
                return {"error": f"Bibcode {bibcode} not found in source data."}

            self.bibcode = bibcode or self.paper.get("bibcode")
            logger.info("Using paper bibcode: %s", self.bibcode)

            # populate the user template with paper data
            self.user_prompt = self.populate_user_template(self.paper)

        logger.info("Submitting prompt...")
        response = self.send_message(user_prompt=self.user_prompt, with_file=with_file)

        # delete the file afterwards
        if with_file:
            # delete the file
            self.client.files.delete(self.file.id)

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

    def create_batch_file(self, bibcodes: list[str]):
        """Create a batch file for the OpenAI Batch API

        Creates a batch file for the OpenAI Batch API. The batch file is a JSONL
        file with each line containing a json of the required inputs for submitting a
        paper to the v1/responses API.

        Parameters
        ----------
        bibcodes : list[str]
            A list of bibcodes to create the batch file for.

        Returns
        -------
        str
            the path to the created batch file
        """
        data = []
        for bibcode in bibcodes:
            paper = get_source(bibcode)
            if not paper:
                logger.warning("No paper source found for bibcode: %s", bibcode)
                continue

            user = get_llm_prompt("user")
            user = self.populate_user_template(paper)
            data.append(
                {
                    "custom_id": bibcode,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": config.llms.openai.model,
                        "instructions": get_llm_prompt("agent"),
                        "input": user,
                        "text": {"format": type_to_text_format_param(InfoModel)},
                    },
                }
            )
        logger.info("Processed %s of %s bibcodes for the batch run", len(data), len(bibcodes))

        # setup the output file
        out = pathlib.Path(config.paths.output) / f"llms/openai_{config.llms.openai.model}/{config.llms.batch_file}"
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        return str(out)

    def submit_batch(self, bibcodes: list[str] = None, batch_file: str = None):
        """Submit a batch of papers for processing

        This method submits a batch of papers for processing by the OpenAI API.
        It can either take a list of bibcodes or a path to a pre-made batch jsonl file.

        Parameters
        ----------
        bibcodes : list[str], optional
            the bibcodes to process, by default None
        batch_file : str, optional
            the path of the jsonl batch file, by default None
        """
        # create a batch file if not provided
        if not batch_file:
            batch_file = self.create_batch_file(bibcodes)

        # upload the batch file
        self.upload_file(file_path=batch_file, purpose="batch")

        # submit the batch job
        self.batch = self.client.batches.create(
            endpoint="/v1/responses", input_file_id=self.file.id, completion_window="24h"
        )
        logger.info("Submitting a batch run with batch ID %s", self.batch.id)

    def retrieve_batch(self, batch_id: str = None):
        """Retrieve a batch of papers for processing

        This method retrieves the status and results of a batch job submitted
        to the OpenAI API. Upon job completion, it extracts the output, formats it, and
        writes it out to config.llms.prompt_output_file.

        Parameters
        ----------
        batch_id : str, optional
            the id of the submitted batch job, by default None

        Raises
        ------
        openai.APIStatusError
            when the batch run has failed
        """
        # get the batch id
        batch_id = batch_id or self.batch.id if self.batch else None
        if not batch_id and not self.batch:
            self.batch = next(iter(self.client.batches.list()), None)
            batch_id = batch_id or self.batch.id if self.batch else None

        # retrieve the batch
        self.batch = self.client.batches.retrieve(batch_id)

        if self.batch.status in {"failed", "expired", "canceled", "cancelling"}:
            raise openai.APIStatusError(f"Batch submission did not complete, with status: {self.batch.status}")

        if self.batch.status == "in_progress":
            logger.info("Batch run is still in progress. Please wait and try again later.")
            return

        if self.batch.status == "completed":
            file_response = self.client.files.content(self.batch.output_file_id)

            # parse the response and format to our standard output
            data = {}
            for line in file_response.response.iter_lines():
                i = json.loads(line)
                ii = InfoModel(**json.loads(i["response"]["body"]["output"][0]["content"][0]["text"]))
                data[i["custom_id"]] = [ii.model_dump()]

            # write the output to our standard llm output file
            out = (
                pathlib.Path(config.paths.output)
                / f"llms/openai_{config.llms.openai.model}/{config.llms.prompt_output_file}"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Writing batch output to %s", out)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)


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
    verbose: bool = None,
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
    verbose : bool, optional
        Flag to turn on verbose logging, by default None
    """
    oa = OpenAIHelper(verbose=verbose)

    # iterate for number of runs
    for i in range(n_runs):
        # submit the paper to the LLM
        response = oa.submit_paper(filepath=file_path, bibcode=bibcode, index=index)

        # log the prompts if verbosity set
        if oa.verbose:
            logger.info("Agent Prompt: %s", oa.agent_prompt)
            logger.info("User Prompt: %s", oa.user_prompt)
            logger.info("Original Prompt Response: %s", oa.original_response)

        logger.info("Output: %s", response)

        # write the output response to a file
        key = oa.get_output_key()
        write_output(key, response)

