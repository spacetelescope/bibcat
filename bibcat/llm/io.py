
import json
import os
import tempfile

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

