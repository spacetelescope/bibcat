

import argparse
import json
import os
import pathlib

import openai
from openai import OpenAI

from bibcat import config

# Define your system prompt here
SYSTEM_PROMPT = """You are an expert researcher and professor in astronomy, with many years of experience. You
understand the academic paper.  Your job is to read papers and extract information about the observational datasets.

In addition, you should always report two things: one, the primary mission or survey the dataset used in the paper comes
from, and two, any other mission or survey that has been mentioned in the paper.  Every request should come with a
new paper attached.  If not prompt the user to attach the paper and wait for them.

Always structure your response the same for every paper, as a JSON response with the following fields:
"title": the title of the paper; "summary": a one-line summary of the paper"; "primary_mission_or_survey": a list of primary
missions or surveys the dataset used in the paper comes from, with one item per mission or survey;
"other_missions_or_surveys_mentioned": a list of other missions or surveys mentioned
in the paper, with one item per mission or survey; and "notes", any other miscellaneous notes or thoughts you have. If no
observational dataset can be identified then say so, put it in the "notes" section, and leave those fields blank.
Always include the title though. Don't make up answers. Anything else you have to say should go in the "notes" section, and not
outside the json context. You're a great astronomer and thorough reader who likes to be correct."""


# set up the OpenAI API client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


# Function to read the file content
def read_file(file_path: str) -> bytes:
    with open(file_path, 'rb') as file:
        return file.read()


# upload the file
def upload_file(file_path: str) -> str:
    response = client.files.create(
        file=open(file_path, "rb"),
        purpose='assistants'
    )
    return response.id



def create_vector_store(name: str = 'Papers', files: list = None) -> str:
    vector_store = client.beta.vector_stores.create(
        name=name,
        files=files
    )
    return vector_store


def get_vector_store(vs_id: str) -> str:
    try:
        vector_store = client.beta.vector_stores.retrieve(vs_id)
    except openai.NotFoundError:
        return None
    else:
        return vector_store

#vector_store = client.beta.vector_stores.create(name="Papers")
#vector_store = client.beta.vector_stores.retrieve("vs_WQUCrr0Nw9jBOA5sFu7YurnE")



def create_assistant(vs_id: str = None):
    assistant = client.beta.assistants.create(
        name="Paper Reader",
	instructions=SYSTEM_PROMPT,
        model=config.llms.openai.model,
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vs_id]}}
    )
    return assistant


def assistant_request(file_id: str) -> dict:

    # main paper reader, with file_search
    assistant = client.beta.assistants.retrieve('asst_85Ay7apSmArPOMKF4iRY893b')

    # create a thread
    thread = client.beta.threads.create(
        messages=[
           {"role": "user", "content": "What is this dataset?",
            "attachments": [
               { "file_id": file_id, "tools": [{"type": "file_search"}] }],
           }
        ]
    )

    # submit the prompt request
    print(thread.tool_resources.file_search)
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id=assistant.id
    )

    # extract the response
    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
    message_content = messages[0].content[0].text
    print('message', message_content.value)
    response =  message_content.value.strip('```').strip('json')

    # do some cleanup; delete the file and the temporary vector store
    vs = thread.tool_resources.file_search.vector_store_ids[0]
    client.beta.vector_stores.delete(vs)
    client.files.delete(file_id)

    return json.loads(response)


def write_output(filepath, response):

    path = pathlib.Path(filepath)

    # setup the output file
    out = pathlib.Path(config.paths.output) / f'llms/openai_{config.llms.openai.model}/paper_output.json'
    out.parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(out):
        data = {path.name: [response]}
        with open(out, 'w+') as f:
            json.dump(data, f, indent=2, sort_keys=False)
    else:
        with open(out, 'r') as f:
            data = json.load(f)
        if path.name in data:
            data[path.name].append(response)
        else:
            data[path.name] = [response]
        with open(out, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=False)


def run(file_path: str, run: int = 1):
    for i in range(run):
        file_id = upload_file(file_path)
        print(file_id)
        response = assistant_request(file_id)
        print(response)

        write_output(file_path, response)