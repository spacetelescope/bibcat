
import json
import pathlib

import pandas as pd

from bibcat import config
from bibcat.llm.io import get_source
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)


def read_output(bibcode: str = None) -> list:
    """ Read in the output for a given bibcode

    Returns the content from the output JSON file
    for the given bibcode.

    Parameters
    ----------
    bibcode : str, optional
        The paper bibcode, by default None

    Returns
    -------
    list
        The output data from the LLM response
    """
    out = pathlib.Path(config.paths.output) / f'llms/openai_{config.llms.openai.model}/{config.llms.prompt_output_file}'

    with open(out, 'r') as f:
        data = json.load(f)
        return data.get(bibcode) if bibcode else data



def evaluate_output(bibcode: str = None, index: int = None) -> pd.DataFrame:
    """ Evaluate the output from the LLM model

    For a given paper bibcode, reads in the output from the LLM model and
    evaluates its performance against the human paper classifications. It matches
    the LLM's predicted mission and papertype against the human classification, and
    computes a cursory accuracy score based on the number of runs. It also provides
    a flag indicating whether the LLM mission + papertype was in the set of human
    classification.  It logs to the console and prints which prediced missions are missing by
    the humans, and which human missions are missing by the LLM.

    Returns a pandas DataFrame grouped by LLM mission and paper type, with columns for
    its mean confidence score, the count of this entry in the LLM output, the total number
    of trial runs, the accuracy of the LLM classification, and a flag indicating whether
    the LLM classification was in the human classification.

    Parameters
    ----------
    bibcode : str, optional
        the paper bibcode, by default None
    index : int, optional
        the dataset array index, by default None

    Returns
    -------
    pd.DataFrame
        an output pandas dataframe
    """
    # get the output
    paper = get_source(bibcode=bibcode, index=index)
    bibcode = paper['bibcode']
    response = read_output(bibcode)

    # exit if no bibcode found in output
    if not response:
        logger.info(f'No output found for {bibcode}')
        return None

    n_runs = len(response)

    logger.info(f'Evaluating output for {bibcode}')
    logger.info(f'Number of runs: {n_runs}')

    # convert output to a dataframe
    df = pd.DataFrame([(k, v[0], v[1]) for i in response for k, v in i.items()],
                      columns=['mission', 'papertype', 'llm_confidence'])
    df = df.sort_values('mission').reset_index(drop=True)

    # group by mission and paper type,
    # get the mean confidence and the count in each group
    grouped_df = df.groupby(['mission', 'papertype']).\
        agg(mean_llm_confidence=('llm_confidence', 'mean'), count=('llm_confidence', 'size')).reset_index().\
            rename(columns={'mission':'llm_mission', 'papertype':'llm_papertype'})
    grouped_df['n_runs'] = n_runs

    # get the human paper classifications
    human_classes = paper['class_missions']
    formatted_output = "\n".join([f"{mission}: {info['papertype']}" for mission, info in human_classes.items()])
    logger.info(f"Human Classifications:\n {formatted_output}")

    # compute accuracy of matches to human classification
    vv = [(k, v['papertype']) for k, v in human_classes.items()]
    grouped_df['accuracy'] = grouped_df.apply(lambda x: (x['count']/x['n_runs']) * 100 if (x['llm_mission'], x['llm_papertype']) in vv else 0, axis=1)
    grouped_df['in_human_class'] = grouped_df.apply(lambda x: (x['llm_mission'], x['llm_papertype']) in vv, axis=1)

    # get missing missions
    missing_by_human = set(grouped_df['llm_mission']) - set(human_classes)
    missing_by_llm = set(human_classes)-set(grouped_df['llm_mission'])

    # log the output
    logger.info("Output Stats by LLM Mission and Paper Type:\n" + grouped_df.to_string(index=False))
    logger.info('Missing missions by humans: ' + ', '.join(missing_by_human))
    logger.info('Missing missions by LLM: ' + ', '.join(missing_by_llm))

    return grouped_df



