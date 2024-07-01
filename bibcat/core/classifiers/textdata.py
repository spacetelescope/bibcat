"""
:title: textdata.py

This module is a class purely meant to be inherited by the various *Classifier classes and inherits Base() class.
In short, the ClassifierBase class is a collection of methods used by different classifier types.

The primary methods and use cases of ClassifierBase include:
* `classify_text`: Base classification method, overwritten by various classifier types during inheritance.
* `_process_text`: Use the Grammar class (and internally the Paper class) to process given text into modifs.

"""
from bibcat.core.base import Base
from bibcat.core.grammar import Grammar


# TODO - remove this ; not needed
class ClassifierBase(Base):
    """
    WARNING! This class is *not* meant to be used directly by users.
    -
    Class: ClassifierBase
    Purpose:
     - Container for common underlying methods used in *Classifier classes.
     - Purely meant to be inherited by *Classifier classes.
    -
    """

    # Initialize this class instance
    def __init__(self):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initializes instance of ClassifierBase class.
        """
        # Nothing to see here - inheritance base
        pass

    # Base classification; overwritten by inheritance as needed
    def classify_text(self, text):
        """
        Method: classify_text
        WARNING! This method is a placeholder for inheritance by other classes.
        """
        # Nothing to see here - inheritance base
        pass

    # Load text and process into modifs using Grammar class
    # TODO - remove this ; this is for rules
    def _process_text(self, text, keyword_obj, which_mode, do_check_truematch, buffer=0, do_verbose=False):
        """
        Method: _process_text
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Process text into modifs using Grammar class.
        """

        # Generate and store instance of Grammar class for this text
        use_these_modes = list(set([which_mode, "none"]))
        grammar = Grammar(
            text, keyword_obj=keyword_obj, do_check_truematch=do_check_truematch, do_verbose=do_verbose, buffer=buffer
        )
        grammar.run_modifications(which_modes=use_these_modes)
        self._store_info(grammar, "grammar")

        # Fetch modifs and grammar information
        set_info = grammar.get_modifs(which_modes=[which_mode], do_include_forest=True)
        modif = set_info["modifs"][which_mode]
        forest = set_info["_forest"]

        # Return all processed statements
        return {"modif": modif, "forest": forest}
