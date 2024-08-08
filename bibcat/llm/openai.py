
import argparse
import json
import os
import pathlib
import re

import openai
from openai import OpenAI

from bibcat import config
from bibcat.llm.io import get_file, get_llm_prompt

# set up the OpenAI API client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))



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
        instructions=get_llm_prompt('agent'),
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
           {"role": "user", "content": get_llm_prompt('user'),
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

    # get the response content
    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
    message_content = messages[0].content[0].text
    print('message', message_content.value)
    response = check_response(message_content.value)

    # do some cleanup; delete the file and the temporary vector store
    vs = thread.tool_resources.file_search.vector_store_ids[0]
    client.beta.vector_stores.delete(vs)
    client.files.delete(file_id)

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
    # extract the json content
    response = re.search(r'```json\n(.*?)\n```', value, re.DOTALL)

    if response:
        response = response.group(1)
        return json.loads(response)
    else:
        return {'error': 'No JSON content found in response'}


def write_output(filepath, response):

    # get the filename ; for sources, use the bibcode
    path = pathlib.Path(filepath)
    name = path.name
    if name.startswith('temp'):
        name = name.rsplit('_',1)[-1].split('.json')[0]

    # setup the output file
    out = pathlib.Path(config.paths.output) / f'llms/openai_{config.llms.openai.model}/paper_output.json'
    out.parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(out):
        data = {name: [response]}
        with open(out, 'w+') as f:
            json.dump(data, f, indent=2, sort_keys=False)
    else:
        with open(out, 'r') as f:
            data = json.load(f)
        if name in data:
            data[name].append(response)
        else:
            data[name] = [response]
        with open(out, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=False)


def run(file_path: str = None, bibcode: str = None, index: int = None, run: int = 1):
    for i in range(run):
        file_path = get_file(filepath=file_path, bibcode=bibcode, index=index)

        file_id = upload_file(file_path)
        print(file_id)

        response = assistant_request(file_id)
        print(response)

        write_output(file_path, response)