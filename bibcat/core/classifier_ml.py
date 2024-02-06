"""
:title: classifier_ml.py

"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
from core.classfier import _Classifier
from official.nlp import optimization as tf_opt

import bibcat.config as config


class Classifier_ML(_Classifier):
    """
    Class: Classifier_ML
    Purpose:
        - Train a machine learning model on text within a directory.
        - Use a trained machine learning model to classify given text.
    Initialization Arguments:
        - class_names [list of str]:
          - Names of the classes used in classification.
        - filepath_model [None or str (default=None)]:
          - Filepath of a model to load, or None to load no models.
        - fileloc_model [None or str (default=None)]:
          - File folder location of model-related information to load, or None to load no information.
        - do_verbose [bool (default=False)]:
          - Whether or not to print surface-level log information and tests.
    """

    # Initialize this class instance
    def __init__(self, filepath_model=None, fileloc_ML=None, do_verbose=False):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initializes instance of Classifier_ML class.
        """

        # Store information about this instance
        self._storage = {}  # Dictionary to hold all information
        # Load model and related information, if given
        if filepath_model is not None:
            load_dict = np.load(filepath_model, allow_pickle=True).item()

            class_names = load_dict["class_names"]
            optimizer = tf_opt.create_optimizer(
                init_lr=load_dict["init_lr"],
                num_train_steps=load_dict["num_steps_train"],
                num_warmup_steps=load_dict["num_steps_warmup"],
                optimizer_type=load_dict["type_optimizer"],
            )
            model = tf.keras.models.load_model(fileloc_ML, custom_objects={config.ML_name_optimizer: optimizer})

        # Otherwise, store empty placeholder
        else:
            model = None
            load_dict = None
            class_names = None

        # Store the model and related quantities
        self._store_info(class_names, "class_names")
        self._store_info(model, "model")
        self._store_info(load_dict, "dict_info")
        self._store_info(do_verbose, "do_verbose")

        return

    # Build an empty ML model
    def _build_ML(self, ml_preprocessor, ml_encoder, frac_dropout, num_dense, activation_dense):
        """
        Method: _build_ML
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Build an empty, layered machine learning (ML) model.
        """

        # Assemble the layers for the empty model
        # NOTE: Structure is:
        # =Text Input -> Preprocessor -> Encoder -> Dropout layer -> Dense layer
        # Text input
        layer_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
        # Preprocessor
        layer_preprocessor = tfhub.KerasLayer(ml_preprocessor, name="preprocessor")
        # Encoder
        inputs_encoder = layer_preprocessor(layer_input)
        layer_encoder = tfhub.KerasLayer(ml_encoder, trainable=True, name="encoder")
        outputs_encoder = layer_encoder(inputs_encoder)

        # Construct the overall model
        net = outputs_encoder["pooled_output"]
        net = tf.keras.layers.Dropout(frac_dropout)(net)
        net = tf.keras.layers.Dense(num_dense, activation=activation_dense, name="classifier")(net)

        # Return the completed empty model
        return tf.keras.Model(layer_input, net)

    # Train and save an empty ML model
    def train_ML(self, dir_model, name_model, seed, do_verbose=None, do_return_model=False):
        """
        Method: train_ML
        Purpose: Build an empty machine learning (ML) model and train it.
        Arguments:
          - dir_model [str]:
            - File location containing directories of training, validation, and testing data text entries. Model will be saved here.
          - name_model [str]:
            - Base name for this model.
          - do_save [bool (default=False)]:
            - Whether or not to save model and related output.
          - seed [int]:
            - Seed for random number generation.
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
        Returns:
          - dict:
            - 'model': the model itself.
            - 'dict_history': outputs from model training.
            - 'accuracy': accuracy from model training.
            - 'loss': loss from model training.
        """
        # Load global variables
        dir_train = os.path.join(dir_model, config.folders_TVT["train"])
        dir_validation = os.path.join(dir_model, config.folders_TVT["validate"])
        dir_test = os.path.join(dir_model, config.folders_TVT["test"])

        savename_ML = config.tfoutput_prefix + name_model
        savename_model = name_model + ".npy"

        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")

        # Load in ML values
        label_mode = config.ML_label_model
        batch_size = config.ML_batch_size
        type_optimizer = config.ML_type_optimizer
        ml_model_key = config.ML_model_key
        frac_dropout = config.ML_frac_dropout
        frac_steps_warmup = config.ML_frac_steps_warmup
        num_epochs = config.ML_num_epochs
        init_lr = config.ML_init_lr
        activation_dense = config.ML_activation_dense

        # Throw error if model already exists
        if os.path.exists(os.path.join(dir_model, savename_model)):
            raise ValueError(
                "Err: Model already exists, will not overwrite." + "\n{0}, at {1}.".format(savename_model, dir_model)
            )

        elif os.path.exists(os.path.join(dir_model, savename_ML)):
            raise ValueError(
                "Err: ML output already exists, will not overwrite" + ".\n{0}, at {1}.".format(savename_ML, dir_model)
            )

        # Load in the training, validation, and testing datasets
        if do_verbose:
            print("Loading datasets...")
            print("Loading training data...")

        # For training
        dataset_train_raw = tf.keras.preprocessing.text_dataset_from_directory(
            dir_train, batch_size=batch_size, label_mode=label_mode, seed=seed
        )
        class_names = dataset_train_raw.class_names
        num_dense = len(class_names)
        dataset_train = dataset_train_raw.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        # For validation
        if do_verbose:
            print("Loading validation data...")
        dataset_validation = (
            tf.keras.preprocessing.text_dataset_from_directory(
                dir_validation, batch_size=batch_size, label_mode=label_mode, seed=seed
            )
            .cache()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        # For testing
        if do_verbose:
            print("Loading testing data...")
        dataset_test = (
            tf.keras.preprocessing.text_dataset_from_directory(
                dir_test, batch_size=batch_size, label_mode=label_mode, seed=seed
            )
            .cache()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        # Print some notes
        if do_verbose:
            print("Done loading datasets.")

        # Load in the ML model components
        if do_verbose:
            print("Loading ML model components...")

        # Load the preprocessor
        ml_handle_preprocessor = config.dict_ml_model_preprocessors[ml_model_key]
        ml_preprocessor = tfhub.KerasLayer(ml_handle_preprocessor)
        if do_verbose:
            print("Loaded ML preprocessor: {0}".format(ml_handle_preprocessor))

        ml_handle_encoder = config.dict_ml_model_encoders[ml_model_key]
        ml_encoder = tfhub.KerasLayer(ml_handle_encoder, trainable=True)
        if do_verbose:
            print("Loaded ML encoder: {0}".format(ml_handle_encoder))
            print("Done loading ML model components.")

        # Build an empty ML model
        if do_verbose:
            print("Building an empty ML model...")
            print("Dropout fraction: {0}".format(frac_dropout))
            print("Number of Dense layers: {0}".format(num_dense))

        model = self._build_ML(
            ml_preprocessor=ml_preprocessor,
            ml_encoder=ml_encoder,
            frac_dropout=frac_dropout,
            num_dense=num_dense,
            activation_dense=activation_dense,
        )

        # Print some notes
        if do_verbose:
            print("Done building an empty ML model.")

        # Set up the loss, metric, and optimization functions
        if do_verbose:
            print("Setting up loss, metric, and optimization functions...")

        init_loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = [tf.keras.metrics.CategoricalAccuracy("accuracy")]
        stepsize_epoch = tf.data.experimental.cardinality(dataset_train).numpy()

        num_steps_train = stepsize_epoch * num_epochs
        num_steps_warmup = int(frac_steps_warmup * num_steps_train)

        optimizer = tf_opt.create_optimizer(
            init_lr=init_lr,
            num_train_steps=num_steps_train,
            num_warmup_steps=num_steps_warmup,
            optimizer_type=type_optimizer,
        )

        # Print some notes
        if do_verbose:
            print("# of training steps: {0}\n# of warmup steps: {1}".format(num_steps_train, num_steps_warmup))
            print("Type of optimizer and initial lr: {0}, {1}".format(type_optimizer, init_lr))

        # Compile the model with the loss, metric, and optimization functions
        model.compile(optimizer=optimizer, loss=init_loss, metrics=metrics)
        if do_verbose:
            print("Done compiling loss, metric, and optimization functions.")
            print(model.summary())

        # Run and evaluate the model on the training and validation data
        if do_verbose:
            print("\nTraining the ML model...")
        history = model.fit(x=dataset_train, validation_data=dataset_validation, epochs=num_epochs)

        if do_verbose:
            print("\nTesting the ML model...")
        res_loss, res_accuracy = model.evaluate(dataset_test)

        if do_verbose:
            print("\nDone training and testing the ML model!")

        # Save the model
        save_dict = {
            "loss": res_loss,
            "class_names": class_names,
            "accuracy": res_accuracy,
            "init_lr": init_lr,
            "num_epochs": num_epochs,
            "num_steps_train": num_steps_train,
            "num_steps_warmup": num_steps_warmup,
            "type_optimizer": type_optimizer,
        }
        model.save(os.path.join(dir_model, savename_ML), include_optimizer=False)
        np.save(os.path.join(dir_model, savename_model), save_dict)

        # Plot the results
        self._plot_ML(model=model, history=history.history, dict_info=save_dict, folder_save=dir_model)

        # Below Section: Exit the method
        if do_verbose:
            print("\nTraining complete.")

        if do_return_model:
            return model
        else:
            return

    # Run trained ML model on given text
    def _run_ML(self, model, texts, do_verbose=False):
        """
        Method: _run_ML
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Use trained model to classify given text.
        """

        # Run the model on the given texts
        results = model.predict(texts)

        # Print some notes
        if do_verbose:
            for ii in range(0, len(texts)):
                print("{0}:\n{1}\n".format(texts[ii], results[ii]))

        return results

    # Plot structure and results of ML model
    def _plot_ML(self, model, history, dict_info, folder_save):
        """
        Method: _plot_ML
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Plot recorded loss, accuracy, etc. for a trained model.
        """

        # Extract variables
        num_epochs = dict_info["num_epochs"]

        # For plot of loss and accuracy over time
        # For base plot
        e_arr = np.arange(0, num_epochs, 1)
        fig = plt.figure(figsize=(10, 10))
        plt.tight_layout()

        # For loss
        ax = plt.subplot(2, 1, 1)
        ax.set_title("Test loss: {0}\nTest accuracy: {1}".format(dict_info["loss"], dict_info["accuracy"]))
        ax.plot(e_arr, history["loss"], label="Loss: Training", color="blue", linewidth=4)
        ax.plot(e_arr, history["val_loss"], label="Loss: Validation", color="gray", linewidth=2)
        leg = ax.legend(loc="best", frameon=False)
        leg.set_alpha(0.5)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")

        # For accuracy
        ax = plt.subplot(2, 1, 2)
        ax.plot(e_arr, history["accuracy"], label="Accuracy: Training", color="blue", linewidth=4)
        ax.plot(e_arr, history["val_accuracy"], label="Accuracy: Validation", color="gray", linewidth=2)
        leg = ax.legend(loc="best", frameon=False)
        leg.set_alpha(0.5)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")

        # Save and close the plot
        plt.savefig(os.path.join(folder_save, "fig_model_lossandacc.png"))
        plt.close()

        return

    # Classify a single block of text
    def classify_text(self, text, threshold, do_check_truematch=None, keyword_obj=None, do_verbose=False, forest=None):
        """
        Method: classify_text
        Purpose: Classify given text using stored machine learning (ML) model.
        Arguments:
          - forest [None (default=None)]:
            - Unused - merely an empty placeholder for uniformity of classify_text across Classifier_* classes. Keep as None.
          - keyword_objs [list of Keyword instances, or None (default=None)]:
            - List of Keyword instances for which previously constructed paragraphs will be extracted.
          - threshold [str]:
            - The minimum uncertainty allowed to return a classification.
          - text [str]:
            - The text to classify.
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
        Returns:
          - dict:
            - 'verdict': the classification.
            - 'scores_comb': the final score.
            - 'scores_indiv': the individual scores.
            - 'uncertainty': the uncertainty of the classification.
        """

        # Load global variables
        list_classes = self._get_info("class_names")  # dict_info")["class_names"]
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")

        # Print some notes
        if do_verbose:
            print("\nRunning classify_text for ML classifier:")
            print("Class names from model:\n{0}\n".format(list_classes))

        # Cleanse the text
        text_clean = self._streamline_phrase(text)

        # Fetch and use stored model
        model = self._get_info("model")
        probs = np.asarray(model.predict([text_clean]))[0]  # Uncertainties
        dict_uncertainty = {list_classes[ii]: probs[ii] for ii in range(0, len(list_classes))}  # Dict. version

        # Determine best verdict
        max_ind = np.argmax(probs)
        max_verdict = list_classes[max_ind]

        # Return low-uncertainty verdict if below given threshold
        if (threshold is not None) and (probs[max_ind] < threshold):
            dict_results = config.dictverdict_lowprob.copy()
            dict_results["uncertainty"] = dict_uncertainty

        # Otherwise, generate dictionary of results
        else:
            dict_results = {
                "verdict": max_verdict,
                "scores_comb": None,
                "scores_indiv": None,
                "uncertainty": dict_uncertainty,
            }

        # Print some notes
        if do_verbose:
            print("\nMethod classify_text for ML classifier complete!")
            print("Max verdict: {0}\n".format(max_verdict))

        # Return dictionary of results
        return dict_results
