"""
:title: textdata.py

This module is a class purely meant to be inherited by the various *Classifier classes and inherits Base() class.
In short, the ClassifierBase class is a collection of methods used by different classifier types.

The primary methods and use cases of ClassifierBase include:
* `classify_text`: Base classification method, overwritten by various classifier types during inheritance.

"""
from bibcat.core.base import Base
from bibcat.core.grammar import Grammar


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
