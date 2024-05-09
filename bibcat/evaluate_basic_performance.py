"""
:title: evaluate_basic_performance.py

This module employs

    1) performance.evaluate_performance_basic() to generates a file of dictionary of the basic performance evaluation information and a plot of a confusion matrix after the classification of input test texts conducted based on a trained ML model or the rule-based model. It script also produces a list of the mis-classified papers.

    2) performance.evaluate_performance_uncertainty() to

All the output are saved in the output folder.

- Basic: generate confusion matrices for the set of Operators
         (containing the different classifiers).
- Uncertainty: plot performance as a function of uncertainty level
               for the set of Operators.


"""

from numpy import float64
from numpy.typing import NDArray

from bibcat.core import operator, performance


def generate_performance_evaluation_output(
    classifier_name: str,
    classifier: object,
    dict_texts: dict,  # this dict is very complex so a proper type annotation can be determined later
    is_text_processed: bool,
    mapper: dict,
    keyword_objs: list,
    mode_modif: str,
    buffer: int,
    threshold: float,
    threshold_array: NDArray[float64],
    print_freq: int,
    filepath_output: str,
    fileroot_evaluation: str,
    fileroot_misclassif: str,
    fileroot_uncertainty_plot: str,
    fileroot_confusion_matrix_plot: str,
    figsize: tuple,
    load_check_truematch: bool = True,
    do_save_evaluation: bool = True,
    do_save_misclassif: bool = True,
    do_raise_innererror: bool = False,
    do_verbose: bool = True,
    do_verbose_deep: bool = False,
):
    # Initialize operators by loading models into instances of the Operator class
    op = operator.Operator(
        classifier=classifier,
        name=classifier_name,
        mode=mode_modif,
        keyword_objs=keyword_objs,
        load_check_truematch=load_check_truematch,
        do_verbose=do_verbose,
        do_verbose_deep=do_verbose_deep,
    )

    # Create an instance of the Performance class
    performer = performance.Performance()

    # Run the pipeline for a basic evaluation of model performance
    performer.evaluate_performance_basic(
        operators=[op],
        dicts_texts=[dict_texts],
        mappers=[mapper],
        thresholds=[threshold],
        buffers=[buffer],
        is_text_processed=is_text_processed,
        filepath_output=filepath_output,
        filename_plot=fileroot_confusion_matrix_plot,
        fileroot_evaluation=fileroot_evaluation,
        fileroot_misclassif=fileroot_misclassif,
        figsize=figsize,
        print_freq=print_freq,
        do_verbose=do_verbose,
        do_verbose_deep=do_verbose_deep,
        do_raise_innererror=do_raise_innererror,
        do_save_evaluation=do_save_evaluation,
        do_save_misclassif=do_save_misclassif,
        do_verify_truematch=load_check_truematch,
    )

    # Run the pipeline for an evaluation of model performance
    # as a function of uncertainty
    performer.evaluate_performance_uncertainty(
        operators=[op],
        dicts_texts=[dict_texts],
        mappers=[mapper],
        threshold_arrays=[threshold_array],
        buffers=[buffer],
        is_text_processed=is_text_processed,
        filepath_output=filepath_output,
        filename_plot=fileroot_uncertainty_plot,
        fileroot_evaluation=fileroot_evaluation,
        fileroot_misclassif=fileroot_misclassif,
        figsize=figsize,
        print_freq=print_freq,
        do_verify_truematch=load_check_truematch,
        do_raise_innererror=do_raise_innererror,
        do_save_evaluation=do_save_evaluation,
        do_save_misclassif=do_save_misclassif,
        do_verbose=do_verbose,
        do_verbose_deep=do_verbose_deep,
    )
