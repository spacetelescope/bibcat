"""
:title: rules.py

"""
import itertools as iterer

import numpy as np
from nltk.corpus import wordnet

from bibcat import config
from bibcat.core.classifiers.textdata import ClassifierBase


class RuleBasedClassifier(ClassifierBase):
    """
    Class: RuleBasedClassifier
    Purpose:
        - Use an internal 'decision tree' to classify given text.
    Initialization Arguments:
        - which_classifs [list of str or None (default=None)]:
          - Names of the classes used in classification. If None, will load from bibcat_constants.py.
        - do_verbose [bool (default=False)]:
          - Whether or not to print surface-level log information and tests.
        - do_verbose_deep [bool (default=False)]:
          - Whether or not to print inner log information and tests.
    """

    # Initialize this class instance
    def __init__(self, which_classifs=None, do_verbose=False, do_verbose_deep=False):
        # Initialize storage
        self._storage = {}
        # Store global variables
        self._store_info(do_verbose, "do_verbose")
        self._store_info(do_verbose_deep, "do_verbose_deep")
        if which_classifs is None:
            which_classifs = config.grammar.list_default_verdicts_decisiontree
        self._store_info(which_classifs, "class_names")

        # Assemble the fixed decision tree
        decision_tree = self._assemble_decision_tree()
        self._store_info(decision_tree, "decision_tree")

        # Print some notes
        if do_verbose:
            print("> Initialized instance of RuleBasedClassifier class.")
            print("Internal decision tree has been assembled.")
            print("NOTE: Decision tree probabilities:\n{0}\n".format(decision_tree))

        return

    # Apply a decision tree to a 'nest' dictionary for some text
    def _apply_decision_tree(self, decision_tree, tree_nest):
        # Extract global variables
        do_verbose = self._get_info("do_verbose")
        which_classifs = self._get_info("class_names")
        keys_main = config.grammar.rules.nest_keys_main
        keys_matter = [item for item in keys_main if (item.endswith("matter"))]
        bool_keyword = "is_keyword"
        prefix = "prob_"

        dict_nest = tree_nest.copy()
        for key in keys_main:  # Encapsulate any single values into tuples
            if isinstance(dict_nest[key], str) or (dict_nest[key] is None):
                dict_nest[key] = [tree_nest[key]]
            else:
                dict_nest[key] = tree_nest[key]

        # Print some notes
        if do_verbose:
            print("Applying decision tree to the following nest:")
            print(dict_nest)

        # Reject this nest if no keywords within
        if not any([bool_keyword in dict_nest[key] for key in keys_matter]):
            # Print some notes
            if do_verbose:
                print("No keywords remaining for this cleaned nest. Skipping.")
            return None

        # Find matching decision tree branch
        best_branch = None
        for key_tree in decision_tree:
            curr_branch = decision_tree[key_tree]
            # Determine if current branch matches
            is_match = True
            # Iterate through parameters
            for key_param in keys_main:
                # Skip ahead if current parameter allows any value ('is_any')
                if curr_branch[key_param] == "is_any":
                    continue

                # Otherwise, check for exact matching values based on branch form
                elif isinstance(curr_branch[key_param], tuple):
                    # Check for exact matching values
                    if not np.array_equal(np.sort(curr_branch[key_param]), np.sort(dict_nest[key_param])):
                        is_match = False
                        break  # Exit from this branch early

                # Otherwise, check inclusive and excluded ('!') matching values
                elif isinstance(curr_branch[key_param], list):
                    # Check for included matching values
                    if (
                        not all(
                            [
                                (item in dict_nest[key_param])
                                for item in curr_branch[key_param]
                                if (not item.startswith("!"))
                            ]
                        )
                    ) or (
                        any(
                            [
                                (item in dict_nest[key_param])
                                for item in curr_branch[key_param]
                                if (item.startswith("!"))
                            ]
                        )
                    ):
                        is_match = False
                        break  # Exit from this branch early

                # Otherwise, check if any of allowed values contained
                elif isinstance(curr_branch[key_param], set):
                    # Check for any of matching values
                    if not any([(item in dict_nest[key_param]) for item in curr_branch[key_param]]):
                        is_match = False
                        break  # Exit from this branch early

                # Otherwise, throw error if format not recognized
                else:
                    raise ValueError("Err: Invalid format for {0}!".format(curr_branch))

            # Store this branch as match, if valid
            if is_match:
                best_branch = curr_branch
                # Print some notes
                if do_verbose:
                    print("\n- Found matching decision branch:\n{0}".format(best_branch))

                break
            # Otherwise, carry on
            else:
                pass

        # Raise an error if no matching branch found
        if (not is_match) or (best_branch is None):
            raise ValueError("Err: No match found for {0}!".format(dict_nest))

        # Extract the probabilities from the branch
        dict_probs = {key: best_branch[prefix + key] for key in which_classifs}

        # Return the final probabilities
        if do_verbose:
            print("Final scores computed for nest:\n{0}\n\n{1}\n\n{2}".format(dict_nest, dict_probs, best_branch))

        return {"probs": dict_probs, "components": best_branch}

    # Assemble base of decision tree, with probabilities, that can be read from/expanded as full decision tree
    def _assemble_decision_tree(self):
        # Extract global variables
        do_verbose = self._get_info("do_verbose")
        which_classifs = self._get_info("class_names")
        dict_possible_values = config.grammar.dict_tree_possible_values
        # dict_valid_combos = config.dict_tree_valid_value_combinations
        keys_matter = config.grammar.rules.nest_keys_matter
        key_verbtype = config.grammar.rules.nest_key_verbtype
        all_params = list(dict_possible_values.keys())
        prefix = "prob_"

        # Goal: Generate matrix of probabilities based on 'true' examples
        if True:  # Just to make it easier to hide the example content display
            dict_examples_base = {}
            itrack = -1

            # <know verb classes>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": "is_any",
                "objectmatter": "is_any",
                "verbclass": {"know"},
                "verbtypes": "is_any",
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 1.0,
            }

            # <be,has verb classes>
            # 'OBJ data has/are/had/were available in the archive.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_keyword"]),
                "objectmatter": tuple([]),
                "verbclass": {"be", "has"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 1.0,
            }

            # 'The stars have/are/had/were available in OBJ data.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple([]),
                "objectmatter": tuple(["is_keyword"]),
                "verbclass": {"be", "has"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 1.0,
            }

            # 'We have/had OBJ data/Our rms is/was small for the OBJ data.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_1st"]),
                "objectmatter": tuple(["is_keyword"]),
                "verbclass": {"be", "has"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 1.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # ’The data is/was from OBJ by Authorsetal.’
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple([]),
                "objectmatter": tuple(["is_keyword", "is_etal"]),
                "verbclass": {"be", "has"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.5,
            }

            # ’The data is from OBJ by Authorsetal.’
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple([]),
                "objectmatter": tuple(["is_keyword", "is_etal"]),
                "verbclass": {"be"},
                "verbtypes": {"PRESENT"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.8,
                "prob_mention": 0.6,
            }

            # ’We know/knew the limits of the OBJ data.’
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_1st"]),
                "objectmatter": tuple(["is_keyword"]),
                "verbclass": {"know"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.5,
            }

            # ’!.’
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple([]),
                "objectmatter": tuple(["is_keyword", "is_pron_1st", "is_etal"]),
                "verbclass": {"know"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.25,
                "prob_mention": 0.5,
            }

            # <science,plot verb classes>
            # 'OBJ data shows/detects/showed/detected a trend.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_keyword"]),
                "objectmatter": tuple([]),
                "verbclass": {"plot", "science"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 0.4,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.6,
            }

            # 'People use/used OBJ data.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple([]),
                "objectmatter": tuple(["is_keyword"]),
                "verbclass": {"plot", "science"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 0.4,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.6,
            }

            # 'We detect/plot/detected/plotted OBJ data.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_1st"]),
                "objectmatter": tuple(["is_keyword"]),
                "verbclass": {"plot", "science"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 1.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # 'This work detects/plots our OBJ data.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple([]),
                "objectmatter": tuple(["is_pron_1st", "is_keyword"]),
                "verbclass": {"plot", "science"},
                "verbtypes": {"PRESENT"},
                "prob_science": 1.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # 'Figures 1-10 detect/plot/detected/plotted OBJ data.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_term_fig"]),
                "objectmatter": tuple(["is_keyword"]),
                "verbclass": {"plot", "science"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 1.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # 'This work detects/plots the OBJ data in Figure 1.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple([]),
                "objectmatter": tuple(["is_term_fig", "is_keyword"]),
                "verbclass": {"plot", "science"},
                "verbtypes": {"PRESENT"},
                "prob_science": 1.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # 'We detect/plot/detected/plotted the OBJ data in Figure 1.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_1st"]),
                "objectmatter": tuple(["is_term_fig", "is_keyword"]),
                "verbclass": {"plot", "science"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 1.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # 'Authorsetal detect/plot/detected/plotted OBJ data.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_etal"]),
                "objectmatter": tuple(["is_keyword"]),
                "verbclass": {"plot", "science"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.1,
                "prob_mention": 1.0,
            }

            # 'Authorsetal detect/plot/detected/plotted the OBJ data in Fig 1.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_etal"]),
                "objectmatter": tuple(["is_term_fig", "is_keyword"]),
                "verbclass": {"plot", "science"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 0.0,
                "prob_data_influenced": 1.0,
                "prob_mention": 0.5,
            }

            # 'The data shows trends for the OBJ data by Authorsetal in Fig 1.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple([]),
                "objectmatter": tuple(["is_term_fig", "is_keyword", "is_etal"]),
                "verbclass": {"plot", "science"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 0.0,
                "prob_data_influenced": 1.0,
                "prob_mention": 0.5,
            }

            # 'The trend uses/plots/used/plotted the OBJ data from Authorsetal.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple([]),
                "objectmatter": tuple(["is_etal", "is_keyword"]),
                "verbclass": {"plot", "science"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 0.0,
                "prob_data_influenced": 1.0,
                "prob_mention": 0.5,
            }

            # 'Their OBJ observations detects/detected the star.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_keyword", "is_pron_3rd"]),
                "objectmatter": tuple([]),
                "verbclass": {"science"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.2,
                "prob_mention": 0.6,
            }

            # 'They plot/plotted/detect/detected stars in their OBJ data.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_3rd"]),
                "objectmatter": tuple(["is_keyword"]),
                "verbclass": {"science", "plot"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.2,
                "prob_mention": 0.6,
            }

            # 'Their OBJ observations detects/detected the star.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_keyword", "is_pron_3rd"]),
                "objectmatter": tuple([]),
                "verbclass": {"science", "plot"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.2,
                "prob_mention": 0.6,
            }

            # <Data-influenced stuff>
            # ’We simulate/simulated the OBJ data.’
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_1st"]),
                "objectmatter": tuple(["is_keyword"]),
                "verbclass": {"datainfluenced"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.8,
                "prob_mention": 0.4,
            }

            # ’We simulate/simulated the OBJ data of Authorsetal.’
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_1st"]),
                "objectmatter": tuple(["is_keyword", "is_etal"]),
                "verbclass": {"datainfluenced"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.8,
                "prob_mention": 0.6,
            }

            # ’Authorsetal simulate/simulated OBJ data in their study.’
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_etal"]),
                "objectmatter": tuple(["is_keyword", "is_pron_3rd"]),
                "verbclass": {"datainfluenced"},
                "verbtypes": {"PRESENT", "PAST"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 1.0,
            }

            # <All FUTURE verbs>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": "is_any",
                "objectmatter": "is_any",
                "verbclass": "is_any",
                "verbtypes": ["FUTURE"],
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 1.0,
            }

            # <All POTENTIAL verbs>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": "is_any",
                "objectmatter": "is_any",
                "verbclass": "is_any",
                "verbtypes": ["POTENTIAL"],
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 1.0,
            }

            # <All PURPOSE verbs>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": "is_any",
                "objectmatter": "is_any",
                "verbclass": "is_any",
                "verbtypes": ["PURPOSE"],
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 1.0,
            }

            # <All None verbs>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": "is_any",
                "objectmatter": "is_any",
                "verbclass": {None},
                "verbtypes": "is_any",
                "prob_science": 0.5,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

            # <All nonverb verbs - usually captions>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": "is_any",
                "objectmatter": ["is_etal"],
                "verbclass": {"root_nonverb"},
                "verbtypes": "is_any",
                "prob_science": 0.0,
                "prob_data_influenced": 0.8,
                "prob_mention": 0.5,
            }

            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": ["is_etal"],
                "objectmatter": "is_any",
                "verbclass": {"root_nonverb"},
                "verbtypes": "is_any",
                "prob_science": 0.0,
                "prob_data_influenced": 0.8,
                "prob_mention": 0.5,
            }

            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": "is_any",
                "objectmatter": "is_any",
                "verbclass": {"root_nonverb"},
                "verbtypes": "is_any",
                "prob_science": 0.8,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

            # Stuff from missing branch output, now with '!' ability: 2023-05-30
            # is_key/is_proI/is_fig; is_etal/is_they
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": ["is_pron_1st"],
                "objectmatter": {"is_etal", "is_pron_3rd"},
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

            # is_key/is_proI/is_fig; is_proI, !is_etal, !is_they
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": {"is_keyword", "is_pron_1st", "is_term_fig"},
                "objectmatter": ["is_pron_1st", "!is_etal", "!is_pron_3rd"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 1.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # is_key; is_fig, !is_etal, !is_they
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_keyword"]),
                "objectmatter": ["is_term_fig", "!is_etal", "!is_pron_3rd"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 1.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # is_key; is_etal, !is_proI, !is_fig
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_keyword"]),
                "objectmatter": ["is_etal", "!is_pron_1st", "!is_term_fig"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.5,
            }

            # is_key; is_they, !is_proI, !is_fig
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_keyword"]),
                "objectmatter": ["is_pron_3rd", "!is_pron_1st", "!is_term_fig"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

            # is_fig; is_etal, !is_pron_1st
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_term_fig"]),
                "objectmatter": ["is_etal", "!is_pron_1st"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

            # is_fig, !is_etal, !is_they; is_etal, !is_pron_1st
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": ["is_term_fig", "!is_etal", "!is_pron_3rd"],
                "objectmatter": ["is_keyword", "!is_etal", "!is_pron_3rd"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 1.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # is_etal + is_any (not proI, fig), is_any (not proI, fig)
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": ["is_etal", "!is_pron_1st", "!is_term_fig"],
                "objectmatter": ["!is_pron_1st", "!is_term_fig"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 1.0,
            }

            # is_they + is_any (not proI, fig), is_any (not proI, fig)
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": ["is_pron_3rd", "!is_pron_1st", "!is_term_fig"],
                "objectmatter": ["!is_pron_1st", "!is_term_fig"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 1.0,
            }

            # Stuff from missing branch output: 2023-05-25
            # Missing single stuff:
            # is_key; is_key
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_keyword"]),
                "objectmatter": tuple(["is_keyword"]),
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 1.0,
            }

            # is_key; is_fig
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_keyword"]),
                "objectmatter": tuple(["is_term_fig"]),
                "verbclass": {"be", "has", "plot", "science", "datainfluenced"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 1.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # is_etal; is_key
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_etal"]),
                "objectmatter": tuple(["is_keyword"]),
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 1.0,
            }

            # Multi-combos and subj.obj. duplicates
            # is_key; is_proI, is_key
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_keyword"]),
                "objectmatter": tuple(["is_pron_1st", "is_keyword"]),
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # is_key; is_proI, is_fig
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_keyword"]),
                "objectmatter": tuple(["is_pron_1st", "is_term_fig"]),
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # is_key; is_proI, is_they + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_keyword"]),
                "objectmatter": ["is_pron_1st", "is_pron_3rd"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

            # is_key; is_proI, is_etal + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_keyword"]),
                "objectmatter": ["is_pron_1st", "is_etal"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

            # is_key; is_proI, is_etal + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_keyword"]),
                "objectmatter": ["is_pron_1st", "is_etal"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

            # is_key; is_etal, is_key
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_keyword"]),
                "objectmatter": tuple(["is_keyword", "is_etal"]),
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.5,
            }

            # is_key; is_etal, is_fig
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_keyword"]),
                "objectmatter": tuple(["is_term_fig", "is_etal"]),
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.5,
            }

            # is_key; is_etal, is_they
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_keyword"]),
                "objectmatter": tuple(["is_pron_3rd", "is_etal"]),
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.5,
            }

            # is_proI; is_proI, is_key
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_1st"]),
                "objectmatter": tuple(["is_pron_1st", "is_keyword"]),
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # is_proI; is_etal, is_key
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_1st"]),
                "objectmatter": tuple(["is_etal", "is_keyword"]),
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

            # is_proI; is_they, is_key
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_1st"]),
                "objectmatter": tuple(["is_pron_3rd", "is_keyword"]),
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

            # is_proI; is_fig, is_key
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_1st"]),
                "objectmatter": tuple(["is_term_fig", "is_keyword"]),
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # is_proI; is_proI, is_etal, is_key + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_1st"]),
                "objectmatter": ["is_pron_1st", "is_keyword", "is_etal"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

            # is_proI; is_proI, is_they, is_key + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_1st"]),
                "objectmatter": ["is_pron_1st", "is_keyword", "is_pron_3rd"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

            # is_proI; is_proI, is_fig, is_key + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_1st"]),
                "objectmatter": ["is_pron_1st", "is_term_fig", "is_keyword"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # is_proI; is_etal, is_they, is_key + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_1st"]),
                "objectmatter": ["is_etal", "is_keyword", "is_pron_3rd"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

            # is_proI; is_etal, is_fig, is_key + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_1st"]),
                "objectmatter": ["is_etal", "is_keyword", "is_term_fig"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

            # is_proI; is_they, is_fig, is_key + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_1st"]),
                "objectmatter": ["is_pron_3rd", "is_keyword", "is_term_fig"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

            # is_etal; is_key + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_etal"]),
                "objectmatter": ["is_keyword"],  # tuple(["is_pron_1st", "is_keyword"]),
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.5,
            }

            # is_they; is_key + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_3rd"]),
                "objectmatter": ["is_keyword"],  # tuple(["is_pron_1st", "is_keyword"]),
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

            # Stuff with empty subj/obj.matter: 2023-05-31
            # is_keyword, is_proI; None; plot, science
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_1st", "is_keyword"]),
                "objectmatter": tuple([]),
                "verbclass": {"plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # None; is_keyword; data-infl.; no-verbtype
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple(["is_pron_1st"]),
                "objectmatter": tuple(["is_keyword"]),
                "verbclass": {"plot"},  # {"be", "has", "plot", "science"},
                "verbtypes": tuple([]),
                "prob_science": 0.5,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # None; is_keyword; data-infl.; no-verbtype
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple([]),
                "objectmatter": tuple(["is_keyword"]),
                "verbclass": {"datainfluenced"},  # {"be", "has", "plot", "science"},
                "verbtypes": tuple([]),
                "prob_science": 0.0,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.0,
            }

            # None; is_keyword; !data-infl.; no-verbtype
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple([]),
                "objectmatter": tuple(["is_keyword"]),
                "verbclass": ["!datainfluenced"],  # {"be", "has", "plot", "science"},
                "verbtypes": tuple([]),
                "prob_science": 0.5,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

            # None; is_etal, !is_proI, !is_fig; no-verbtype
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": tuple([]),
                "objectmatter": ["is_etal", "!is_pron_1st", "!is_term_fig"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": tuple([]),
                "prob_science": 0.0,
                "prob_data_influenced": 0.0,
                "prob_mention": 1.0,
            }

            # !is_etal, !is_they; is_proI, !is_etal, !is_they
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": ["!is_etal", "!is_pron_3rd"],
                "objectmatter": ["is_pron_1st", "!is_etal", "!is_pron_3rd"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # !is_etal, !is_they; is_proI, !is_etal, !is_they
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": ["!is_etal", "!is_pron_3rd"],
                "objectmatter": ["is_term_fig", "!is_etal", "!is_pron_3rd"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.0,
                "prob_mention": 0.0,
            }

            # !is_etal, !is_they; is_proI, !is_etal, !is_they
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": ["!is_etal", "!is_pron_3rd"],
                "objectmatter": ["is_keyword", "!is_etal", "!is_pron_3rd"],
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": {"PAST", "PRESENT"},
                "prob_science": 0.5,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

            # No-verbtype, all-combos
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter": "is_any",
                "objectmatter": "is_any",
                "verbclass": "is_any",  # {"be", "has", "plot", "science"},
                "verbtypes": tuple([]),
                "prob_science": 0.5,
                "prob_data_influenced": 0.5,
                "prob_mention": 0.5,
            }

        # Generate final base tree with only target classifs and norm. probs.
        decision_tree = {}
        itrack = -1
        # Iterate through base examples
        for key_ex in dict_examples_base:
            curr_ex = dict_examples_base[key_ex]
            curr_denom = np.sum([curr_ex[(prefix + item)] for item in which_classifs])  # Prob. normalizer
            curr_probs = {(prefix + item): (curr_ex[(prefix + item)] / curr_denom) for item in which_classifs}

            # For main example
            # Extract all parameters and their values
            new_ex = {key: curr_ex[key] for key in all_params}
            # Normalize and store probabilities for target classifs
            new_ex.update(curr_probs)
            # Store this example
            itrack += 1
            decision_tree[itrack] = new_ex

            # For passive example
            # For general (not is_any, not set) case
            if curr_ex[key_verbtype] == "is_any":
                #                    or (curr_ex[key_verbtype] is None)):
                pass  # No passive counterpart necessary for is_any case
            # For set, add example+passive for each entry in set
            elif isinstance(curr_ex[key_verbtype], set):
                # Iterate through set entries
                for curr_val in curr_ex[key_verbtype]:
                    # Passive with flipped subj-obj
                    # Extract all parameters and their values
                    new_ex = {key: curr_ex[key] for key in all_params if (key not in (keys_matter + [key_verbtype]))}
                    # Add in passive term for verbtypes
                    tmp_vals = [curr_val, "PASSIVE"]
                    new_ex[key_verbtype] = tmp_vals
                    # Flip the subject and object terms
                    new_ex["subjectmatter"] = curr_ex["objectmatter"]
                    new_ex["objectmatter"] = curr_ex["subjectmatter"]
                    # Normalize and store probabilities for target classifs
                    new_ex.update(curr_probs)
                    # Store this example
                    itrack += 1
                    decision_tree[itrack] = new_ex
            else:
                # Extract all parameters and their values
                new_ex = {key: curr_ex[key] for key in all_params if (key not in (keys_matter + [key_verbtype]))}
                # Add in passive term for verbtypes
                tmp_vals = list(curr_ex[key_verbtype]) + ["PASSIVE"]
                # Apply old data structure type to new expanded verbtype
                new_ex[key_verbtype] = type(curr_ex[key_verbtype])(tmp_vals)
                # Flip the subject and object terms
                new_ex["subjectmatter"] = curr_ex["objectmatter"]
                new_ex["objectmatter"] = curr_ex["subjectmatter"]
                # Normalize and store probabilities for target classifs
                new_ex.update(curr_probs)
                # Store this example
                itrack += 1
                decision_tree[itrack] = new_ex

        # Return the assembled decision tree
        return decision_tree

    # Categorize topic of given verb
    def _categorize_verb(self, i_verb, struct_words):
        ##Extract global variables
        verb = struct_words[i_verb]["word"]
        verb_dep = struct_words[i_verb]["_dep"]
        verb_pos = struct_words[i_verb]["_pos"]
        do_verbose = self._get_info("do_verbose")
        list_category_names = config.grammar.rules.list_category_names
        list_category_synsets = config.grammar.rules.list_category_synsets
        list_category_threses = config.grammar.rules.list_category_threses
        max_hyp = config.grammar.rules.max_num_hypernyms
        # root_hypernyms = wordnet.synsets(verb, pos=wordnet.VERB)
        if max_hyp is None:
            root_hypernyms = wordnet.synsets(verb, pos=wordnet.VERB)
        else:
            root_hypernyms = wordnet.synsets(verb, pos=wordnet.VERB)[0:max_hyp]
        num_categories = len(list_category_synsets)

        # Print some notes
        if do_verbose:
            print("\n> Running _categorize_verb().")
            print(
                "Verb: {0}\nMax #hyp: {1}\nRoot hyp: {2}\nCategories: {3}\n".format(
                    verb, max_hyp, root_hypernyms, list_category_names
                )
            )

        # Handle non-verb roots
        if (verb_dep in ["ROOT"]) and (verb_pos in ["NOUN"]):
            if do_verbose:
                print("Verb {0} is a root noun. Marking as such.")

            return config.grammar.rules.category_nonverb_root

        # Handle specialty verbs
        # For 'be' verbs
        if any([(roothyp in config.grammar.rules.synsets_verbs_be) for roothyp in root_hypernyms]):
            return "be"
        # For 'has' verbs
        elif any([(roothyp in config.grammar.rules.synsets_verbs_has) for roothyp in root_hypernyms]):
            return "has"

        # Determine likely topical category for this verb
        score_alls = [None] * num_categories
        score_fins = [None] * num_categories
        pass_bools = [None] * num_categories
        # Iterate through the categories
        for ii in range(0, num_categories):
            score_alls[ii] = [
                roothyp.path_similarity(mainverb)
                for mainverb in list_category_synsets[ii]
                for roothyp in root_hypernyms
            ]
            # Take max score, if present
            if len(score_alls[ii]) > 0:
                score_fins[ii] = max(score_alls[ii])
            else:
                score_fins[ii] = 0
            # Determine if this score passes any category thresholds
            pass_bools[ii] = score_fins[ii] >= list_category_threses[ii]

        # Throw an error if no categories fit this verb well
        if not any(pass_bools):
            if do_verbose:
                print("No categories fit verb: {0}, {1}\n".format(verb, score_fins))
            return None

        # Throw an error if this verb gives very similar top scores
        thres = config.grammar.rules.thres_category_fracdiff
        metric_close_raw = np.abs(np.diff(np.sort(score_fins)[::-1])) / max(score_fins)
        metric_close = metric_close_raw[0]
        if metric_close < thres:
            # Select most extreme verb with the max score
            tmp_max = max(score_fins)
            if score_fins[list_category_names.index("plot")] == tmp_max:
                tmp_extreme = "plot"
            elif score_fins[list_category_names.index("science")] == tmp_max:
                tmp_extreme = "science"
            else:
                raise ValueError("Reconsider extreme categories for scoring!")

            # Print some notes
            if do_verbose:
                print(
                    "Multiple categories with max score: {0}: {1}\n{2}\n{3}".format(
                        verb, root_hypernyms, score_fins, list_category_names
                    )
                )
                print("Selecting most extreme verb: {0}\n".format(tmp_extreme))
            # Return the selected most-extreme score
            return tmp_extreme

        # Return the determined topical category with the best score
        best_category = list_category_names[np.argmax(score_fins)]
        # Print some notes
        if do_verbose:
            print("Best category: {0}\nScores: {1}".format(best_category, score_fins))
        # Return the best category
        return best_category

    # Classify a set of statements (rule approach)
    def _classify_statements(self, forest, threshold, do_verbose=None):
        # Extract global variables
        if do_verbose is not None:  # Override do_verbose if specified for now
            self._store_info(do_verbose, "do_verbose")
        # Load the fixed decision tree
        decision_tree = self._get_info("decision_tree")
        # Print some notes
        if do_verbose:
            print("\n> Running _classify_statements.")
            print("Forest word-trees:")
            for key in forest:
                print("{0}".format(forest[key]["modestruct_words_info"].keys()))
                print("{0}\n".format(forest[key]["modestruct_words_info"]))

        # Set the booleans for this statement dictionary
        forest_nests = self._make_nest_forest(forest)["main"]
        num_trees = len(forest_nests)

        # Send each statement through the decision tree
        dict_scores = []  # None]*num_trees
        list_comps = []  # None]*num_trees
        for ii in range(0, num_trees):
            # Split this nest into unlinked components
            nests_unlinked = self._unlink_nest(forest_nests[ii])
            # Compute score for each component
            curr_scores = []  # [None]*len(nests_unlinked)
            curr_comps = []
            for jj in range(0, len(nests_unlinked)):
                tmp_stuff = self._apply_decision_tree(decision_tree=decision_tree, tree_nest=nests_unlinked[jj])
                if tmp_stuff is not None:  # Store if not None
                    curr_scores.append(tmp_stuff["probs"])
                    curr_comps.append(tmp_stuff["components"])

            # Combine the scores
            if len(curr_scores) > 0:
                dict_scores.append(self._combine_unlinked_scores(curr_scores))
                list_comps.append(curr_comps)

        # Convert the tree scores into a set of verdicts
        resdict = self._convert_scorestoverdict(
            dict_scores_indiv=dict_scores, components=list_comps, threshold=threshold
        )

        # Return the dictionary containing verdict, etc. for these statements
        return resdict

    # Classify full text based on its statements (rule approach)
    def classify_text(
        self,
        keyword_obj,
        threshold,
        do_check_truematch,
        which_mode=None,
        forest=None,
        text=None,
        buffer=0,
        do_verbose=None,
    ):
        # Fetch global variables
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")
        do_verbose_deep = self._get_info("do_verbose_deep")
        # Store/Override latest keyword object and None return of classif.
        self._store_info(keyword_obj, "keyword_obj")

        # Process the text into paragraphs and their statements
        if forest is None:
            forest = self._process_text(
                text=text,
                do_check_truematch=do_check_truematch,
                keyword_obj=keyword_obj,
                do_verbose=do_verbose_deep,
                buffer=buffer,
                which_mode=which_mode,
            )["forest"]

        # Extract verdict dictionary of statements for keyword object
        dict_results = self._classify_statements(forest, do_verbose=do_verbose, threshold=threshold)

        # Print some notes
        if do_verbose:
            print("Verdicts complete.")
            print("Verdict dictionary:\n{0}".format(dict_results))
            print("---")

        # Return final verdicts
        return dict_results  # dict_verdicts

    # Add together scores from unlinked components of a nest
    def _combine_unlinked_scores(self, component_scores):
        # Extract global variables
        do_verbose = self._get_info("do_verbose")
        # Print some notes
        if do_verbose:
            print("\n> Running _combine_unlinked_scores.")
            print("Considering set of scores:\n{0}".format(component_scores))

        # Verify all score keys are the same
        if any([(component_scores[0].keys() != item.keys()) for item in component_scores]):
            raise ValueError("Err: Unequal score keys?\n{0}".format(component_scores))

        # Combine the scores across all components
        keys_score = list(component_scores[0].keys())
        fin_scores = {key: 0 for key in keys_score}
        tot_score = 0  # Total score for normalization purposes
        for ii in range(0, len(component_scores)):
            for key in component_scores[ii]:
                fin_scores[key] += component_scores[ii][key]
                tot_score += component_scores[ii][key]

        # Normalize the combined scores
        for key in fin_scores:
            if tot_score == 0:  # If empty score, record as 0
                fin_scores[key] = 0
            else:  # Otherwise, normalize score
                fin_scores[key] /= tot_score

        # Return the combined scores
        if do_verbose:
            print("\n> Run of _combine_unlinked_scores complete!")
            print("Combined scores:\n{0}".format(fin_scores))

        return fin_scores

    # Convert set of decision tree scores into single verdict
    def _convert_scorestoverdict(
        self,
        dict_scores_indiv,
        components,
        threshold,
        max_diff_thres=0.10,
        max_diff_count=3,
        max_diff_verdicts=["science"],
    ):
        # Extract global variables
        do_verbose = self._get_info("do_verbose")
        # Print some notes
        if do_verbose:
            print("\n> Running _convert_scorestoverdict.")
            print("Individual components and score sets:")
            print("Probability threshold: {0}".format(threshold))
            for ii in range(0, len(components)):
                print("{0}\n{1}\n-".format(components[ii], dict_scores_indiv[ii]))

        # Return empty verdict if empty scores
        # For completely empty scores
        if len(dict_scores_indiv) == 0:
            tmp_res = config.ml.dictverdict_error.copy()
            # Print some notes
            if do_verbose:
                print("\n-Empty scores; verdict: {0}".format(tmp_res))

            return tmp_res

        # Otherwise, remove Nones
        dict_scores_indiv = [item for item in dict_scores_indiv if (item is not None)]
        all_keys = list(dict_scores_indiv[0].keys())

        # Calculate and store verdict value statistics from indiv. entries
        num_indiv = len(dict_scores_indiv)
        dict_results = {key: {"score_tot_unnorm": 0, "count_max": 0, "count_tot": num_indiv} for key in all_keys}
        for ii in range(0, num_indiv):
            curr_scores = dict_scores_indiv[ii]
            for curr_key in all_keys:
                tmp_unnorm = curr_scores[curr_key]
                # Determine if current key has max. score across all keys
                if (max_diff_thres is not None) and (tmp_unnorm > 0):
                    tmp_compare = all(
                        [
                            ((np.abs(tmp_unnorm - curr_scores[other_key]) / curr_scores[other_key]) >= max_diff_thres)
                            for other_key in all_keys
                            if (other_key != curr_key)
                        ]
                    )  # Check if rel. max. key by some threshold

                else:
                    tmp_compare = False

                # Increment count of sentences with max-valued verdict
                if tmp_compare:
                    dict_results[curr_key]["count_max"] += 1

                # Increment unnorm. score count
                dict_results[curr_key]["score_tot_unnorm"] += tmp_unnorm

        # Normalize and store the scores
        denom = np.sum([dict_results[key]["score_tot_unnorm"] for key in all_keys])
        for curr_key in all_keys:
            # Calculate and store normalized score
            tmp_score = dict_results[curr_key]["score_tot_unnorm"] / denom
            dict_results[curr_key]["score_tot_norm"] = tmp_score

        list_scores_comb = [dict_results[key]["score_tot_norm"] for key in all_keys]

        # Gather final scores into set of uncertainties
        dict_uncertainties = {key: dict_results[key]["score_tot_norm"] for key in dict_results}

        # Print some notes
        if do_verbose:
            print("Indiv. scores without Nones:\n{0}".format(dict_scores_indiv))
            print("Normalizing denominator: {0}".format(denom))
            print("Full score set:")
            for curr_key in dict_results:
                print("{0}: {1}".format(curr_key, dict_results[curr_key]))
            print("Listed combined scores: {0}: {1}".format(all_keys, list_scores_comb))

        # Determine best verdict and associated probabilistic error
        is_found = False
        # For max sentence count:
        if (not is_found) and (max_diff_thres is not None):
            # Check allowed keys that fit max condition
            for curr_key in max_diff_verdicts:
                if dict_results[curr_key]["count_max"] >= max_diff_count:
                    is_found = True
                    max_score = dict_results[curr_key]["score_tot_norm"]
                    max_verdict = curr_key

                    # Print some notes
                    if do_verbose:
                        print("\n-Max score: {0}".format(max_score))
                        print("Max verdict: {0}\n".format(max_verdict))

                    # Break from loop early if found
                    break

        # For max normalized total score:
        if not is_found:
            max_ind = np.argmax(list_scores_comb)
            max_score = list_scores_comb[max_ind]
            max_verdict = all_keys[max_ind]
            # Print some notes
            if do_verbose:
                print("\n-Max score: {0}".format(max_score))
                print("Max verdict: {0}\n".format(max_verdict))

            # Return verdict only if above given threshold probability
            if (threshold is not None) and (max_score < threshold):
                tmp_res = config.ml.dictverdict_lowprob.copy()
                tmp_res["scores_indiv"] = dict_scores_indiv
                tmp_res["uncertainty"] = dict_uncertainties
                tmp_res["components"] = components
                # Print some notes
                if do_verbose:
                    print("-No scores above probability threshold.")
                    print("Returning low-prob verdict:\n{0}".format(tmp_res))

                return tmp_res

            # Return low-prob verdict if multiple equal top probabilities
            elif list_scores_comb.count(max_score) > 1:
                tmp_res = config.ml.dictverdict_lowprob.copy()
                tmp_res["scores_indiv"] = dict_scores_indiv
                tmp_res["uncertainty"] = dict_uncertainties
                tmp_res["components"] = components
                # Print some notes
                if do_verbose:
                    print("-Multiple top prob. scores.")
                    print("Returning low-prob verdict:\n{0}".format(tmp_res))

                return tmp_res

        # Assemble and return final verdict
        fin_res = {
            "verdict": max_verdict,
            "scores_indiv": dict_scores_indiv,
            "uncertainty": dict_uncertainties,
            "components": components,
        }

        # Print some notes
        if do_verbose:
            print("-Returning final verdict dictionary:\n{0}".format(fin_res))

        return fin_res

    # Determine and return missing branches in decision tree
    def _find_missing_branches(self, do_verbose=None, cap_iter=3, print_freq=100):
        # Load global variables
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")
        decision_tree = self._get_info("decision_tree")

        if do_verbose:
            print("\n\n")
            print(" > Running _find_missing_branches()!")
            print("Loading all possible branch parameters...")

        # Load all possible values
        all_possible_values = config.grammar.dict_tree_possible_values
        solo_values = [None]
        all_subjmatters = [item for item in all_possible_values["subjectmatter"] if (item is not None)]
        all_objmatters = [item for item in all_possible_values["objectmatter"] if (item is not None)]
        all_verbclasses = [item for item in all_possible_values["verbclass"]]  # No shallow copying!
        solo_verbtypes = ["PAST", "PRESENT", "FUTURE"]
        multi_verbtypes = [
            item for item in all_possible_values["verbtypes"] if ((item is not None) and (item not in solo_verbtypes))
        ]

        if do_verbose:
            print("Done loading all possible branch parameters.")
            print("Generating combinations of branch parameters...")

        # Set caps for multi-parameter combinations
        if cap_iter is not None:
            cap_subj = cap_iter
            cap_obj = cap_iter
            cap_vtypes = cap_iter
        else:
            cap_subj = len(all_subjmatters)
            cap_obj = len(all_objmatters)
            cap_vtypes = len(multi_verbtypes)

        # Generate set of all possible individual multi-parameter combinations
        sub_subjmatters = [item for ii in range(1, (cap_subj + 1)) for item in iterer.combinations(all_subjmatters, ii)]
        sub_objmatters = [item for ii in range(1, (cap_obj + 1)) for item in iterer.combinations(all_objmatters, ii)]
        sub_multiverbtypes = [
            item for ii in range(1, (cap_vtypes + 1)) for item in iterer.combinations(multi_verbtypes, ii)
        ]

        sub_allverbtypes = [
            (list(item_sub) + [item_solo]) for item_sub in sub_multiverbtypes for item_solo in solo_verbtypes
        ]  # Fold in verb tense

        # Fold in solo and required values as needed
        sets_verbclasses = all_verbclasses
        sets_subjmatters = sub_subjmatters  # + [None])
        sets_objmatters = sub_objmatters  # + [None])
        sets_allverbtypes = sub_allverbtypes  # + [None])

        if do_verbose:
            print("Combinations of branch parameters complete.")

        # Generate set of all possible branches (all possible valid combos)
        list_branches = [
            {
                "subjectmatter": sets_subjmatters[aa],
                "objectmatter": sets_objmatters[bb],
                "verbclass": sets_verbclasses[cc],
                "verbtypes": sets_allverbtypes[dd],
            }
            for aa in range(0, len(sets_subjmatters))
            for bb in range(0, len(sets_objmatters))
            for cc in range(0, len(sets_verbclasses))
            for dd in range(0, len(sets_allverbtypes))
            if (
                ((sets_subjmatters[aa] is not None) and ("is_keyword" in sets_subjmatters[aa]))
                or ((sets_objmatters[bb] is not None) and ("is_keyword" in sets_objmatters[bb]))
            )  # Must have keyword
        ]

        num_branches = len(list_branches)
        if do_verbose:
            print("{0} branches generated across all parameter combinations.".format(num_branches))
            print("Extracting branches not covered by decision tree...")

        # Collect branches that are missing from decision tree
        bools_is_missing = [None] * num_branches
        for ii in range(0, num_branches):
            curr_branch = list_branches[ii]
            # Apply decision tree and ensure valid output
            try:
                tmp_res = self._apply_decision_tree(decision_tree=decision_tree, tree_nest=curr_branch)

                bools_is_missing[ii] = False  # Branch is covered

            # Record this branch as missing if invalid output
            except:
                # Mark this branch as missing
                bools_is_missing[ii] = True

            # Print some notes
            if do_verbose and ((ii % print_freq) == 0):
                print("{0} of {1} branches have been checked.".format((ii + 1), num_branches))

        # Throw error if any branches not checked
        if None in bools_is_missing:
            raise ValueError("Err: Branches not checked?")

        # Gather and return missing branches
        if do_verbose:
            print("All branches checked.\n{0} of {1} missing.".format(np.sum(bools_is_missing), num_branches))
            print("Run of _find_missing_branches() complete!")

        return np.asarray(list_branches)[np.asarray(bools_is_missing)]

    # Construct nest of bools, etc, for all verb-clauses, all sentences
    def _make_nest_forest(self, forest):
        # Extract global variables
        do_verbose = self._get_info("do_verbose")
        if do_verbose:
            print("\n> Running _make_nest_forest.")

        # Initialize holder for nests
        repr_key = "none"  # list(forest.keys())[0] #"none"
        num_trees = len(forest[repr_key])
        list_nests = [None] * num_trees
        list_nest_main = [None] * num_trees
        # Iterate through trees (sentences)
        for ii in range(0, num_trees):
            curr_struct_verbs = forest[repr_key][ii]["struct_verbs_updated"]
            curr_struct_words = forest[repr_key][ii]["struct_words_updated"]
            num_branches = len(curr_struct_verbs)
            curr_i_verbs = list(curr_struct_verbs.keys())
            curr_i_words = list(curr_struct_words.keys())
            list_nests[ii] = [None] * num_branches
            # Print some notes
            if do_verbose:
                print("\nCurrent tree: {0} ({0}-{1}).".format(ii, num_trees - 1))
                print("\nVerb-tree keys: {0}".format(curr_struct_verbs.keys()))
                print("Verb-tree: {0}\n".format(curr_struct_verbs))
                print("\nWord-tree keys: {0}".format(curr_struct_words.keys()))
                print("Word-tree data: {0}\n".format(curr_struct_words))

            # Iterate through branches (verb-clauses)
            for jj in range(0, num_branches):
                list_nests[ii][jj] = self._make_nest_verbclause(
                    i_verb=curr_i_verbs[jj], struct_verbs=curr_struct_verbs, struct_words=curr_struct_words
                )

            # Pull representative nest for this tree (at tree root)
            tmp_tomax = [
                len(curr_struct_verbs[jj]["i_postverbs"])
                for jj in curr_i_verbs
                if (curr_struct_verbs[jj]["is_important"])
            ]
            if len(tmp_tomax) > 0:
                id_main = np.argmax(tmp_tomax)
            # Otherwise, throw error
            else:
                raise ValueError(
                    "Err: Nothing important:\n{0}\n\n{1}\n\n{2}".format(
                        curr_struct_words[list(curr_struct_words.keys())[0]]["sentence"],
                        curr_struct_verbs,
                        curr_struct_words,
                    )
                )

            list_nest_main[ii] = list_nests[ii][id_main]
            # Print some notes
            if do_verbose:
                print("Branch nests complete.")
                print("Individual nests:")
                for jj in range(0, num_branches):
                    print(list_nests[ii][jj])
                    print("")
                print("\nId of main nest: {0}".format(id_main))

        # Return the completed nests
        return {"all": list_nests, "main": list_nest_main}

    # Construct nest of bools, etc, to describe a branch (verb-clause)
    def _make_nest_verbclause(self, struct_verbs, struct_words, i_verb):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        branch_verb = struct_verbs[i_verb]
        lookup_pos = config.grammar.rules.conv_pos_fromtreetonest
        ignore_pos = config.grammar.rules.nest_unimportant_pos
        target_bools = config.grammar.rules.nest_important_treebools
        # Print some notes
        if do_verbose:
            print("\n> Running _make_nest_verbclause.")
            print("Considering verb branch:\n{0}".format(branch_verb))

        # Build a nest for the given verb
        dict_nest = {
            "i_verb": i_verb,
            "subjectmatter": [],
            "objectmatter": [],
            "verbtypes": branch_verb["verbtype"],
            # "verbclass":self._categorize_verb(branch_verb["verb"].text),
            "verbclass": self._categorize_verb(i_verb=i_verb, struct_words=struct_words),
            "link_verbclass": [],
            "link_verbtypes": [],
            "link_subjectmatter": [],
            "link_objectmatter": [],
        }

        # Iterate through words directly attached to this verb
        tmp_list = list(set((branch_verb["i_branchwords_all"])))
        tmp_list = [item for item in tmp_list if ((item != i_verb) and (struct_words[item]["pos_main"] != "VERB"))]
        for ii in tmp_list:
            # Skip word if not stored (i.e., trimmed word for trimming scheme)
            if ii not in struct_words:
                if do_verbose:
                    print("Word {0} trimmed from word-tree, so skipping...".format(ii))

                continue

            # Pull current word info
            curr_info = struct_words[ii]
            curr_pos_raw = curr_info["pos_main"]
            if do_verbose:
                print("Considering word {0}.".format(curr_info["word"]))
                print("Has info: {0}.".format(curr_info))

            # Skip if unimportant word
            if not curr_info["is_important"]:
                if do_verbose:
                    print("Unimportant word. Skipping.")

                continue

            # Skip if unimportant pos
            if curr_pos_raw in ignore_pos:
                if do_verbose:
                    print("Unimportant pos {0}. Skipping.".format(curr_pos_raw))

                continue

            # Otherwise, convert pos to accepted nest terminology
            try:
                curr_pos = lookup_pos[curr_pos_raw]
            except KeyError:
                # Print context for this error
                print("Upcoming KeyError! Context:")
                print("Word: {0}".format(curr_info["word"]))
                print("Word info: {0}".format(curr_info))
                tmp_sent = np.asarray(list(curr_info["sentence"]))
                print("Sentence: {0}".format(tmp_sent))
                tmp_chunk = struct_words[i_verb]["wordchunk"]
                print("Wordchunk: {0}".format(tmp_chunk))
                print("Chosen main pos.: {0}".format(curr_pos_raw))

                curr_pos = lookup_pos[curr_pos_raw]

            # Store target booleans into this nest
            for is_item in target_bools:
                # If this boolean is True, store if not stored previously
                if curr_info["dict_importance"][is_item] and (is_item not in dict_nest[curr_pos]):
                    dict_nest[curr_pos].append(is_item)
                # Otherwise, pass
                else:
                    pass

        # Print some notes
        if do_verbose:
            print("\nDone iterating through main (not linked) terms.")
            print("Current state of nest:\n{0}\n".format(dict_nest))

        # Iterate through linked verbs
        for vv in branch_verb["i_postverbs"]:
            # Skip verb if not stored (i.e., trimmed for trimming scheme)
            if vv not in struct_words:
                if do_verbose:
                    print("Word {0} trimmed from word-tree, so skipping...".format(vv))

                continue

            # Store current verb class if not stored previously
            dict_nest["link_verbtypes"].append(struct_verbs[vv]["verbtype"])
            link_verbclass = self._categorize_verb(i_verb=struct_verbs[vv]["i_verb"], struct_words=struct_words)
            dict_nest["link_verbclass"].append(link_verbclass)

            # Prepare temporary dictionary to merge with nest dictionary
            tmp_dict = {"link_subjectmatter": [], "link_objectmatter": []}

            # Iterate through words attached to linked verbs
            for ii in struct_verbs[vv]["i_branchwords_all"]:
                # Skip word if not stored (i.e., trimmed for trimming scheme)
                if ii not in struct_words:
                    if do_verbose:
                        print("Word {0} trimmed from word-tree, so skipping...".format(ii))

                    continue

                # Pull current word info
                curr_info = struct_words[ii]
                curr_pos_raw = curr_info["pos_main"]

                # Skip if unimportant word
                if not curr_info["is_important"]:
                    continue

                # Skip if unimportant pos
                if curr_pos_raw in ignore_pos:
                    continue

                # Otherwise, convert pos to accepted nest terminology
                curr_pos = lookup_pos[curr_pos_raw]

                # Store target booleans into this nest
                for is_item in target_bools:
                    curr_key = "link_" + curr_pos
                    # If this boolean is True, store if not stored previously
                    if curr_info["dict_importance"][is_item] and (is_item not in tmp_dict[curr_key]):
                        tmp_dict[curr_key].append(is_item)
                    # Otherwise, pass
                    else:
                        pass

            # Merge the dictionary for this clause into overall nest
            for key in tmp_dict:
                dict_nest[key].append(tmp_dict[key])

        # Return the nest
        if do_verbose:
            print("Nest complete!\n\nVerb branch: {0}\n\nNest: {1}\n".format(branch_verb, dict_nest))
            print("Run of _make_nest_verbclause complete!\n---\n")

        return dict_nest

    # Split a nest into its main and linked components
    def _unlink_nest(self, nest):
        # Extract global variables
        do_verbose = self._get_info("do_verbose")
        keys_main = config.grammar.rules.nest_keys_main
        keys_matter = [item for item in keys_main if (item.endswith("matter"))]
        keys_nonmatter = [item for item in keys_main if (not item.endswith("matter"))]
        key_matter_obj = "objectmatter"
        prefix_link = config.grammar.rules.nest_prefix_link
        terms_superior = config.grammar.rules.nest_important_treebools_superior
        keys_linked_main = [(prefix_link + key) for key in keys_main]
        keys_linked_matter = [(prefix_link + key) for key in keys_matter]
        num_links = len(nest[keys_linked_matter[0]])
        bool_keyword = "is_keyword"
        bool_pronounI = "is_pron_1st"
        # Print some notes
        if do_verbose:
            print("\n> Running _unlink_nest.")
            print("Considering nest:\n{0}\n".format(nest))

        # Extract location of keyword in nest
        matters_with_keyword = [key for key in keys_matter if (bool_keyword in nest[key])]
        matters_with_keyword += [
            key for key in keys_linked_matter if any([(bool_keyword in nest[key][ii]) for ii in range(0, num_links)])
        ]
        matters_with_keyword = [item.replace(prefix_link, "") for item in matters_with_keyword]  # Rem. link mark
        matters_with_keyword = list(set(matters_with_keyword))

        is_proandkey = False

        # Extract and merge components of nest
        components = []
        # Extract main component
        comp_main = {key: nest[key] for key in keys_main}  # Main component
        main_matters = sorted([item for key in keys_matter for item in nest[key]])

        # Note if I-pronoun and keyword already paired
        if (bool_keyword in main_matters) and (bool_pronounI in main_matters):
            is_proandkey = True

        # Merge in any linked components
        if any([(nest[key] not in [[], None]) for key in keys_linked_matter]):
            # Throw error if unequal number of components by matter
            if len(set([len(nest[key]) for key in keys_linked_matter])) != 1:
                raise ValueError(
                    "Err: Unequal num. of matter components?\n{0}".format([nest[key] for key in keys_linked_matter])
                )

            # Extract and merge in each linked component
            for ii in range(0, num_links):
                # Extract current linked component
                curr_matters = sorted([item for key in keys_linked_matter for item in nest[key][ii]])
                # Print some notes
                if do_verbose:
                    print("Current main matters: {0}".format(main_matters))
                    print("Considering for linkage: {0}".format(curr_matters))

                # Skip this component if no interesting terms
                if len(curr_matters) == 0:
                    # Print some notes
                    if do_verbose:
                        print("No interesting terms, so skipping.")

                    continue

                # Copy over keyword if only keyword present
                if bool_keyword in curr_matters:
                    # Tack on keyword, if not done so already
                    if bool_keyword not in main_matters:
                        comp_main[key_matter_obj].append(bool_keyword)
                        main_matters.append(bool_keyword)  # Mark as included

                # Copy over any precedent terms
                for term in curr_matters:
                    if term in terms_superior:
                        if term not in main_matters:
                            comp_main[key_matter_obj].append(term)
                            main_matters.append(term)
                        # Override main terms with non-matter terms, if 'I'-term
                        if (bool_pronounI in curr_matters) and not (is_proandkey):
                            for key in keys_nonmatter:
                                comp_main[key] = nest[(prefix_link + key)][ii]

                # Print some notes
                if do_verbose:
                    print("Done linking current term.")
                    print("Latest main matters: {0}\n".format(main_matters))

        # Store the merged component
        components.append(comp_main)

        # Return the unlinked components of the nest
        if do_verbose:
            print("\nNest has been unlinked!\nComponents: {0}".format(components))
            print("Run of _unlink_nest complete!\n")

        return components
