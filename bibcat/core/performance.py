"""
:title: performance.py

The `Performance` class contains user-friendly methods for estimating the performance
of given classifiers and outputting that performance as, e.g., confusion matrices.
This class can be used after creating the model, training, and saving
a machine learning model.

"""

import os

import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from bibcat import config
from bibcat.core.base import Base


class Performance(Base):
    """
    Class: Performance
    Purpose:
        - !
    Initialization Arguments:
        - !
    """

    def __init__(self, do_verbose=False, do_verbose_deep=False):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initializes instance of Performance class.
        """
        # Initialize storage
        self._storage = {}
        self._store_info(do_verbose, "do_verbose")
        self._store_info(do_verbose_deep, "do_verbose_deep")
        if do_verbose:
            print("Instance of Performance successfully initialized!")

        return

    # Evaluate the basic performance of the internal classifier on a test set of data
    def evaluate_performance_basic(
        self,
        operators,
        dicts_texts,
        mappers,
        thresholds,
        buffers,
        is_text_processed,
        do_check_truematch,
        filepath_output,
        do_raise_innererror,
        do_save_evaluation=False,
        do_save_misclassif=False,
        filename_plot="performance_confmatr_basic.png",
        fileroot_evaluation=None,
        fileroot_misclassif=None,
        figcolor="white",
        figsize=(20, 20),
        fontsize=16,
        hspace=None,
        cmap_abs=plt.cm.BuPu,
        cmap_norm=plt.cm.PuRd,
        print_freq=25,
        do_verbose=None,
        do_verbose_deep=None,
    ):
        """
        Method: evaluate_performance_basic
        Purpose:
          - Evaluate the performance of the internally stored classifier on a test set of data.
        Arguments:
          - !
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
          - do_verbose_deep [bool (default=False)]:
            - Whether or not to print inner log information and tests.
        Returns:
          - dict:
            - !
        """
        # Fetch global variables
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")
        if do_verbose_deep is None:
            do_verbose_deep = self._get_info("do_verbose_deep")
        #
        # Print some notes
        if do_verbose:
            print("\n> Running evaluate_performance_basic()!")
            print("Generating evaluation for the given operators...")
        #

        # Evaluate classifier within each operator
        dict_evaluations = self._generate_evaluation(
            operators=operators,
            dicts_texts=dicts_texts,
            mappers=mappers,
            buffers=buffers,
            is_text_processed=is_text_processed,
            do_check_truematch=do_check_truematch,
            do_raise_innererror=do_raise_innererror,
            do_save_evaluation=do_save_evaluation,
            do_save_misclassif=do_save_misclassif,
            filepath_output=filepath_output,
            fileroot_evaluation=fileroot_evaluation,
            fileroot_misclassif=fileroot_misclassif,
            print_freq=print_freq,
            thresholds=thresholds,
            do_verbose=do_verbose,
            do_verbose_deep=do_verbose_deep,
        )

        # Print some notes
        if do_verbose:
            print("\nEvaluations generated.")
            print("Plotting confusion matrices...")

        # Plot grid of confusion matrices for classifier performance
        titles = [item.name for item in operators]
        list_evaluations = [dict_evaluations[item] for item in titles]
        self.plot_performance_confusion_matrix(
            list_evaluations=list_evaluations,
            list_mappers=mappers,
            list_titles=titles,
            filepath_plot=filepath_output,
            filename_plot=filename_plot,
            figcolor=figcolor,
            figsize=figsize,
            fontsize=fontsize,
            hspace=hspace,
            cmap_abs=cmap_abs,
            cmap_norm=cmap_norm,
        )

        # Print some notes
        if do_verbose:
            print("Confusion matrices have been plotted at:\n{0}".format(filepath_output))

        if do_verbose:
            print("\nRun of evaluate_performance_basic() complete!")

        return

    # Evaluate the performance of the internal classifier on a test set of data as a function of uncertainty
    def evaluate_performance_uncertainty(
        self,
        operators,
        dicts_texts,
        mappers,
        threshold_arrays,
        buffers,
        is_text_processed,
        do_check_truematch,
        filepath_output,
        do_raise_innererror,
        do_save_evaluation=False,
        do_save_misclassif=False,
        filename_plot="performance_grid_uncertainty.png",
        fileroot_evaluation=None,
        fileroot_misclassif=None,
        figcolor="white",
        figsize=(20, 20),
        fontsize=16,
        ticksize=14,
        tickwidth=3,
        tickheight=5,
        colors=["tomato", "dodgerblue", "purple", "dimgray", "silver", "darkgoldenrod", "darkgreen", "green", "cyan"],
        alphas=([0.75] * 10),
        linestyles=["-", "-", "-", "--", "--", "--", ":", ":", ":"],
        linewidths=([3] * 10),
        markers=(["o"] * 10),
        alpha_match=0.5,
        color_match="black",
        linestyle_match="-",
        linewidth_match=8,
        marker_match="*",
        print_freq=25,
        do_verbose=None,
        do_verbose_deep=None,
    ):
        """
        Method: evaluate_performance_uncertainty
        Purpose:
          - Evaluate the performance of the internally stored classifier on a test set
            of data as a function of uncertainty.
        Arguments:
          - !
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
          - do_verbose_deep [bool (default=False)]:
            - Whether or not to print inner log information and tests.
        Returns:
          - dict:
            - !
        """
        # Fetch global variables
        num_ops = len(operators)
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")
        if do_verbose_deep is None:
            do_verbose_deep = self._get_info("do_verbose_deep")

        # Print some notes
        if do_verbose:
            print("\n> Running evaluate_performance_uncertainty()!")
            print("Generating evaluations for operators and uncertainties...")

        # Evaluate classifier for each operator at each uncertainty level
        dict_evaluations = self._generate_evaluation(
            operators=operators,
            dicts_texts=dicts_texts,
            mappers=mappers,
            buffers=buffers,
            is_text_processed=is_text_processed,
            do_check_truematch=do_check_truematch,
            do_raise_innererror=do_raise_innererror,
            do_save_evaluation=do_save_evaluation,
            do_save_misclassif=do_save_misclassif,
            filepath_output=filepath_output,
            fileroot_evaluation=fileroot_evaluation,
            fileroot_misclassif=fileroot_misclassif,
            print_freq=print_freq,
            thresholds=[0] * num_ops,  # Placeholder
            array_thresholds=threshold_arrays,  # Actual uncertainties
            do_verbose=do_verbose,
            do_verbose_deep=do_verbose_deep,
        )

        # Print some notes
        if do_verbose:
            print("\nEvaluations generated.")
            print("Plotting performance as a function of uncertainty level...")

        # Plot grid of classifier performance as function of uncertainty
        titles = [item.name for item in operators]
        list_evaluations = [dict_evaluations[item] for item in titles]
        # Prepare base figure
        fig = plt.figure(figsize=figsize)
        fig.set_facecolor(figcolor)
        nrow = num_ops
        ncol = max([len(item["act_classnames"]) for item in list_evaluations])

        # Iterate through operators (one row per operator)
        for ii in range(0, num_ops):
            curr_xs = threshold_arrays[ii]
            curr_eval = list_evaluations[ii]
            curr_counters = curr_eval["counters"]
            curr_actlabels = sorted(curr_eval["act_classnames"])
            curr_measlabels = sorted(curr_eval["meas_classnames"])
            # Iterate through current actual classifs
            for jj in range(0, len(curr_actlabels)):
                curr_act = curr_actlabels[jj]
                # Prepare subplot
                ax0 = fig.add_subplot(nrow, ncol, ((ii * ncol) + jj + 1))
                # Iterate through measured classifs
                for kk in range(0, len(curr_measlabels)):
                    curr_meas = curr_measlabels[kk]
                    curr_ys = [
                        curr_counters[zz][curr_act][curr_meas] for zz in range(0, len(curr_xs))
                    ]  # Current count of act. vs. meas. classifs
                    # Plot results as function of uncertainty
                    ax0.plot(
                        curr_xs,
                        curr_ys,
                        alpha=alphas[kk],
                        color=colors[kk],
                        linewidth=linewidths[kk],
                        linestyle=linestyles[kk],
                        marker=markers[kk],
                        label=curr_meas,
                    )
                    # Highlight correct answers
                    if curr_act == curr_meas:
                        ax0.plot(
                            curr_xs,
                            curr_ys,
                            alpha=alpha_match,
                            color=colors[kk],
                            linewidth=linewidth_match,
                            linestyle=linestyle_match,
                            marker=marker_match,
                        )

                # Label the subplot
                ax0.set_xlabel("Uncertainty Threshold", fontsize=fontsize)
                ax0.set_ylabel("Count of Classifications", fontsize=fontsize)
                ax0.set_title("{0}: {1} Texts".format(titles[ii], curr_act), fontsize=fontsize)
                ax0.tick_params(width=tickwidth, size=tickheight, labelsize=ticksize, direction="in")

                # Add legend, if last subplot in row
                if jj == (len(curr_actlabels) - 1):
                    ax0.legend(loc="best", frameon=False, prop={"size": fontsize})

        # Save and close the figure
        fig.suptitle("Performance vs. Uncertainty", fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(os.path.join(filepath_output, filename_plot))
        plt.close()

        # Print some notes
        if do_verbose:
            print("Results have been plotted at:\n{0}".format(filepath_output))

        # Exit the method
        if do_verbose:
            print("\nRun of evaluate_performance_uncertainty() complete!")

        return

    # Generate performance evaluation of full classification pipeline (text to rejection/verdict)
    def _generate_evaluation(
        self,
        operators,
        dicts_texts,
        mappers,
        thresholds,
        buffers,
        is_text_processed,
        do_check_truematch,
        do_raise_innererror,
        array_thresholds=None,
        do_save_evaluation=False,
        do_save_misclassif=False,
        filepath_output=None,
        fileroot_evaluation=None,
        fileroot_misclassif=None,
        print_freq=25,
        do_verbose=False,
        do_verbose_deep=False,
    ):
        """
        Method: _generate_evaluation
        Purpose:
          - !
        Arguments:
          - !
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
          - do_verbose_deep [bool (default=False)]:
            - Whether or not to print inner log information and tests.
        Returns:
          - dict:
            - !
        """
        # Fetch global variables
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")
        if do_verbose_deep is None:
            do_verbose_deep = self._get_info("do_verbose_deep")

        num_ops = len(operators)
        # Throw error if operators do not have unique names
        if len(set([item.name for item in operators])) != num_ops:
            raise ValueError(
                "Err: Please give each operator a unique name."
                + "\nCurrently, the names are:\n{0}".format([item.name for item in operators])
            )

        # Print some notes
        if do_verbose:
            print("\n> Running _generate_evaluation()!")
            print("Iterating through Operators to classify each set of text...")

        # Use each operator to classify the set of texts and measure performance
        dict_evaluations = {item.name: None for item in operators}
        for ii in range(0, num_ops):
            curr_op = operators[ii]  # Current operator
            curr_name = curr_op.name
            curr_data = dicts_texts[ii]

            # Print some notes
            if do_verbose:
                print("Classifying with Operator #{0}...".format(ii))

            # Unpack the classified information for this operator
            curr_keys = list(curr_data.keys())  # All keys for accessing texts
            curr_actdicts = [curr_data[curr_keys[jj]] for jj in range(0, len(curr_keys))]  # Forced order
            # Load in as either raw or preprocessed data
            if is_text_processed:  # If given text preprocessed
                curr_texts = None
                curr_modifs = [curr_data[curr_keys[jj]]["text"] for jj in range(0, len(curr_keys))]
            else:  # If given text needs to be preprocessed
                curr_texts = [curr_data[curr_keys[jj]]["text"] for jj in range(0, len(curr_keys))]
                curr_modifs = None

            # Classify texts with current operator
            curr_results = curr_op.classify_set(
                texts=curr_texts,
                modifs=curr_modifs,
                buffer=buffers[ii],
                do_check_truematch=do_check_truematch,
                print_freq=print_freq
                )

            # Print some notes
            if do_verbose:
                print("Classification complete for Operator #{0}.".format(ii))
                print("Generating the performance counter...")

            # Measure performance of current operator against actual answers
            # For standard evaluation
            if array_thresholds is None:  # Store per uncertainty, if given
                tmp_res = self._generate_performance_counter(
                    operator=curr_op,
                    mapper=mappers[ii],
                    list_actdicts=curr_actdicts,
                    list_measdicts=curr_results,
                    print_freq=print_freq,
                    do_verbose=do_verbose,
                    do_verbose_deep=do_verbose_deep,
                )
                curr_counter = tmp_res["counters"]
                curr_misclassifs = tmp_res["misclassifs"]
                curr_meas_classnames = tmp_res["meas_classnames"]
                curr_act_classnames = tmp_res["act_classnames"]

            # For evaluations against multiple uncertainty values
            else:
                tmp_res = [
                    self._generate_performance_counter(
                        operator=curr_op,
                        mapper=mappers[ii],
                        list_actdicts=curr_actdicts,
                        list_measdicts=curr_results,
                        print_freq=print_freq,
                        do_verbose=do_verbose,
                        threshold=array_thresholds[ii][jj],
                        do_verbose_deep=do_verbose_deep,
                    )
                    for jj in range(0, len(array_thresholds[ii]))
                ]
                curr_counter = [item["counters"] for item in tmp_res]
                curr_misclassifs = [item["misclassifs"] for item in tmp_res]
                curr_meas_classnames = tmp_res[0]["meas_classnames"]
                curr_act_classnames = tmp_res[0]["act_classnames"]

            # Print some notes
            if do_verbose:
                print("Performance counter complete.")

            # Store the current results
            dict_evaluations[curr_name] = {
                "counters": curr_counter,
                "misclassifs": curr_misclassifs,
                "actual_results": curr_actdicts,
                "measured_results": curr_results,
                "act_classnames": curr_act_classnames,
                "meas_classnames": curr_meas_classnames,
            }

            # Save the misclassified cases, if so requested
            if do_save_misclassif and (array_thresholds is None):
                # Print some notes
                if do_verbose:
                    print("Saving misclassifications...")
                #
                # Build string of misclassification information
                list_str = [
                    (
                        "Internal ID: {0}\nBibcode: {1}\nMission: {2}\n"
                        + "Actual Classification: {3}\n"
                        + "Measured Classification: {4}\n"
                        + "Modif:\n''\n{5}\n''\n-\n"
                        + "Base Paragraph:\n''\n{6}\n''\n-\n"
                    ).format(
                        curr_misclassifs[key]["id"],
                        curr_misclassifs[key]["bibcode"],
                        curr_misclassifs[key]["mission"],
                        curr_misclassifs[key]["act_classif"],
                        curr_misclassifs[key]["meas_classif"],
                        curr_misclassifs[key]["modif"],
                        curr_misclassifs[key]["modif_none"],
                    )
                    for key in curr_misclassifs
                ]

                str_misclassif = "\n-----\n".join(list_str)  # Combined string

                # Save the full string of misclassifications
                tmp_filename = f"{fileroot_misclassif}.txt"
                tmp_filepath = os.path.join(filepath_output, tmp_filename)
                self._write_text(text=str_misclassif, filepath=tmp_filepath)

                # Print some notes
                if do_verbose:
                    print("\nMisclassifications saved at: {0}".format(tmp_filepath))

            # Print some notes
            if do_verbose:
                print("All work complete for Operator #{0}.".format(ii))

        # Print some notes
        if do_verbose:
            print("!")

        # Save the evaluation components, if so requested
        if do_save_evaluation:
            tmp_filepath = os.path.join(filepath_output, (fileroot_evaluation + ".npy"))
            np.save(tmp_filepath, dict_evaluations)

            # Print some notes
            if do_verbose:
                print("\nEvaluation saved at: {0}".format(tmp_filepath))

        # Return the evaluation components
        if do_verbose:
            print("\nRun of _generate_evaluation() complete!")

        return dict_evaluations

    # Generate performance counter for set of measured classifications vs actual classifications
    def _generate_performance_counter(
        self,
        operator,
        mapper,
        list_actdicts,
        list_measdicts,
        threshold=None,
        print_freq=25,
        do_verbose=False,
        do_verbose_deep=False,
    ):
        """
        Method: _generate_performance_counter
        Purpose:
          - !
        Arguments:
          - !
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
          - do_verbose_deep [bool (default=False)]:
            - Whether or not to print inner log information and tests.
        Returns:
          - dict:
            - !
        """

        # Fetch global variables
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")
        if do_verbose_deep is None:
            do_verbose_deep = self._get_info("do_verbose_deep")

        num_texts = len(list_measdicts)
        meas_classifs = operator.classifier.model.class_names
        # Print some notes
        if do_verbose:
            print("\n> Running _generate_performance_counter()!")

        # Initialize container for counters and misclassifications
        if mapper is not None:  # Use mask classifications, if given
            act_classnames_raw = list(set([item for item in list(mapper.values())]))
            meas_classnames_raw = list(set([item for item in list(mapper.values())]))
        else:  # Otherwise, use internal classifications
            act_classnames_raw = meas_classifs
            meas_classnames_raw = meas_classifs

        # Extend measured allowed class names to include low-uncertainty, etc.
        act_classnames = act_classnames_raw + [config.results.verdict_rejection]
        meas_classnames = meas_classnames_raw + config.results.list_other_verdicts

        # Streamline the class names
        act_classnames = [item.lower().replace("_", "") for item in act_classnames]
        meas_classnames = [item.lower().replace("_", "") for item in meas_classnames]

        # Form containers
        dict_counters = {act_key: {meas_key: 0 for meas_key in meas_classnames} for act_key in act_classnames}
        dict_misclassifs = {}

        # Print some notes
        if do_verbose:
            print("Accumulating performance over {0} texts.".format(num_texts))
            print("Actual class names: {0}\nMeasured class names: {1}".format(act_classnames, meas_classnames))

        # Count up classifications from given texts and classifications
        i_misclassif = 0  # Count of misclassifications
        for ii in range(0, num_texts):
            curr_actdict = list_actdicts[ii]
            curr_measdict = list_measdicts[ii]

            # Iterate through missions that were considered
            for curr_key in curr_measdict:
                lookup = curr_key

                # Extract actual classif
                curr_actval = curr_actdict["missions"][lookup]["class"]
                if (mapper is not None) and (curr_actval.lower() in mapper):
                    # Map to masked value if so requested
                    curr_actval = mapper[curr_actval.lower()]

                # Extract measured classif, or remeasure if threshold given
                curr_measval = curr_measdict[lookup]["verdict"]
                # Remeasure if new threshold given and text was classified
                if (threshold is not None) and (curr_measval.lower().replace("_", "") in act_classnames):
                    tmp_pass = curr_measdict[lookup]["uncertainty"]
                    if tmp_pass is not None:
                        if tmp_pass[curr_measval] < threshold:
                            curr_measval = config.results.dictverdict_lowprob.copy()["verdict"]

                # Map to new masking value, if mapper given
                if (mapper is not None) and (curr_measval.lower() in mapper):
                    curr_measval = mapper[curr_measval.lower()]

                # Streamline the class names
                curr_actval = curr_actval.lower().replace("_", "")
                curr_measval = curr_measval.lower().replace("_", "")

                # Increment current counter
                dict_counters[curr_actval][curr_measval] += 1

                # If misclassification, take note
                if curr_actval != curr_measval:
                    curr_info = {
                        "act_classif": curr_actval,
                        "meas_classif": curr_measval,
                        "bibcode": curr_actdict["bibcode"],
                        "mission": lookup,
                        "id": curr_actdict["id"],
                        "modif": curr_measdict[lookup]["modif"],
                        "modif_none": curr_measdict[lookup]["modif_none"],
                    }

                    dict_misclassifs[str(i_misclassif)] = curr_info
                    # Increment count of misclassifications
                    i_misclassif += 1

        # Compute internal counter totals
        for key_1 in dict_counters:
            curr_sum = sum([dict_counters[key_1][key_2] for key_2 in dict_counters[key_1]])
            dict_counters[key_1]["_total"] = curr_sum

        # Print some notes
        if do_verbose:
            print("\n-\nPerformance counter generated:")
            for key_1 in dict_counters:
                print("Actual {0} total: {1}".format(key_1, dict_counters[key_1]["_total"]))
                for key_2 in dict_counters[key_1]:
                    print("Actual {0} vs Measured {1}: {2}".format(key_1, key_2, dict_counters[key_1][key_2]))

        # Return the counters and misclassifications
        if do_verbose:
            print("\n-\n\nRun of _generate_performance_counter() complete!")

        return {
            "counters": dict_counters,
            "misclassifs": dict_misclassifs,
            "act_classnames": act_classnames,
            "meas_classnames": meas_classnames,
        }

    # Plot confusion matrix for given performance counters
    def plot_performance_confusion_matrix(
        self,
        list_evaluations,
        list_mappers,
        list_titles,
        filepath_plot,
        filename_plot,
        figsize=(20, 6),
        figcolor="white",
        fontsize=16,
        hspace=None,
        cmap_abs=plt.cm.BuPu,
        cmap_norm=plt.cm.PuRd,
        do_verbose=None,
        do_verbose_deep=None,
    ):
        """
        Method: plot_performance_confusion_matrix
        Purpose:
          - !
        Arguments:
          - !
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
          - do_verbose_deep [bool (default=False)]:
            - Whether or not to print inner log information and tests.
        Returns:
          - dict:
            - !
        """

        # Fetch global variables
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")
        if do_verbose_deep is None:
            do_verbose_deep = self._get_info("do_verbose_deep")

        num_evals = len(list_evaluations)
        # Print some notes
        if do_verbose:
            print("\n> Running plot_performance_confusion_matrix()!")

        # Prepare the base figure
        fig = plt.figure(figsize=figsize)
        fig.set_facecolor(figcolor)
        nrow = 2
        ncol = num_evals

        # Plot confusion matrix for each evaluation
        for ii in range(0, num_evals):
            # Use current mapper to determine actual vs measured classifs
            act_classifs = list_evaluations[ii]["act_classnames"]
            meas_classifs = list_evaluations[ii]["meas_classnames"]
            num_act = len(act_classifs)
            num_meas = len(meas_classifs)

            # Initialize container for current confusion matrix
            confmatr_abs = np.zeros(shape=(num_act, num_meas))  # Unnormalized
            confmatr_norm = np.ones(shape=(num_act, num_meas)) * np.nan  # Normalized

            # Fetch counters from current evaluation
            curr_counts = list_evaluations[ii]["counters"]

            # Accumulate the confusion matrices
            for yy in range(0, num_act):  # Iterate through actual classifs
                curr_total = curr_counts[act_classifs[yy]]["_total"]  # Total act.
                for xx in range(0, num_meas):  # Iterate through measured classifs
                    # For the unnormalized confusion matrix
                    curr_val = curr_counts[act_classifs[yy]][meas_classifs[xx]]
                    confmatr_abs[yy, xx] += curr_val

                    # For the normalized confusion matrix
                    confmatr_norm[yy, xx] = confmatr_abs[yy, xx] / curr_total

            # Plot the current set of confusion matrices
            # For the unnormalized matrix
            ax = fig.add_subplot(nrow, ncol, (ii + 1))
            self._ax_confusion_matrix(
                matr=confmatr_abs,
                ax=ax,
                x_labels=meas_classifs,
                y_labels=act_classifs,
                y_title="Actual",
                x_title="Classification",
                cbar_title="Absolute Count",
                ax_title="{0}".format(list_titles[ii]),
                cmap=cmap_abs,
                fontsize=fontsize,
                is_norm=False,
            )

            # For the normalized matrix
            ax = fig.add_subplot(nrow, ncol, (ii + ncol + 1))
            self._ax_confusion_matrix(
                matr=confmatr_norm,
                ax=ax,
                x_labels=meas_classifs,
                y_labels=act_classifs,
                y_title="Actual",
                x_title="Classification",
                cbar_title="Normalized Count",
                ax_title="{0}".format(list_titles[ii]),
                cmap=cmap_norm,
                fontsize=fontsize,
                is_norm=True,
            )

        # Save and close the figure
        plt.tight_layout()
        if hspace is not None:
            plt.subplots_adjust(hspace=hspace)
        plt.savefig(os.path.join(filepath_plot, filename_plot))
        plt.close()

        if do_verbose:
            print("\nRun of plot_performance_confusion_matrix() complete!")

        return
