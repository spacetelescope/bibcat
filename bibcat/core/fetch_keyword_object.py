from typing import Any

from bibcat.utils.logger_config import setup_logger


# Fetch a keyword object that matches the given lookup
def fetch_keyword_object(
    keyword_objs, lookup: str, do_raise_emptyerror: bool = True, verbose: bool = False
) -> Any | None:
    """Fetch a keyword object

    Given an input lookup string, tries to match it to a stored Keyword instance.

    Parameters
    ----------
    lookup : str
        the input string
    do_raise_emptyerror : bool, optional
        Flag to raise an error, by default True

    Returns
    -------
    Any | None
        the matching Keyword instance

    Raises
    ------
    ValueError
        when no match is found
    """

    # Print some notes
    if verbose:
        logger = setup_logger(__name__)
        logger.info(f"> Running _fetch_keyword_object() for lookup term {lookup}.")

    # Find keyword object that matches to given lookup term
    match = None
    for kobj in keyword_objs:
        # If current keyword object matches, record and stop loop
        if kobj.identify_keyword(lookup)["bool"]:
            match = kobj
            break

    # Throw error if no matching keyword object found
    if match is None:
        errstr = f"No matching keyword object for {lookup}.\n"
        errstr += "Available keyword objects are:\n"
        # just use the names of the keywords
        names = ", ".join(a._get_info("name") for a in keyword_objs)
        errstr += f"{names}\n"

        # Raise error if so requested
        if do_raise_emptyerror:
            raise ValueError(errstr)

    # Return the matching keyword object
    return match
