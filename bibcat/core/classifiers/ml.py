import abc
import os

import matplotlib.pyplot as plt
import numpy as np

xxx = None

try:
    import tensorflow as tf  # type: ignore
    import tensorflow_hub as tfhub  # type: ignore
    import tensorflow_text  # type: ignore  # noqa: F401
    from tensorboard.plugins.hparams import api as hp
except ImportError:
    tf = None

from bibcat import config
from bibcat.core.base import Base
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)


class AbstractModel(abc.ABC):
    """ Abstract base class for machine learning models.  To be subclassed by specific model types. """

    @abc.abstractmethod
    def __repr__(self):
        pass

    @abc.abstractmethod
    def build_model(self):
        pass

    @abc.abstractmethod
    def split_datasets(self):
        pass

    @abc.abstractmethod
    def train_model(self):
        pass

    @abc.abstractmethod
    def evaluate_model(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    @abc.abstractmethod
    def save_model(self):
        pass

    @abc.abstractmethod
    def load_model(self):
        pass


class TensorFlow(AbstractModel):

    def __new__(cls, *args, **kwargs):
        if not tf:
            raise ImportError('tensorflow packages not found.  Cannot create a tensorflow model.  Please install required packages.')
        return super(TensorFlow, cls).__new__(cls)

    def __init__(self, model_type: str = 'bert', verbose: bool = False, load: bool = False):
        # object attributes
        self.model_type = model_type
        self.mlconfig = config.ml.get(model_type, {})
        self.model_key = config.ml.ML_model_key
        self.loaded = False
        self.verbose = verbose

        # input and output paths
        name_model = config.output.name_model
        self.data_dir = os.path.join(config.paths.partitioned, name_model)
        self.model_dir = os.path.join(config.paths.models, name_model)
        self.savename_ML = config.output.tfoutput_prefix + name_model
        self.savename_model = name_model + ".npy"
        self.filepath_model = os.path.join(self.model_dir, self.savename_model)
        self.filepath_ML = os.path.join(self.model_dir, self.savename_ML)

        # dataset splitting parameters
        self.label_mode = config.ml.ML_label_model
        self.batch_size = config.ml.ML_batch_size
        self.seed = config.ml.seed_ML

        # ml parameters and attributes
        self.class_names = None
        self.num_dense = None
        self.init_lr = config.ml.ML_init_lr
        self.num_epochs = config.ml.ML_num_epochs
        self.num_steps_train = None
        self.num_steps_warmup = None
        self.type_optimizer = config.ml.ML_type_optimizer

        # model and outputs
        self.model = None
        self.history = None
        self.res_loss = None
        self.res_accuracy = None
        self.outputs = None
        self.hparams = None

        # load the existing model if specified
        if load:
            self.load_model()

    def __repr__(self) -> str:
        return f"<TensorFlow (model_type='{self.model_type}', key='{self.model_key}', loaded={self.loaded})>"

    def build_model(self, num_dense: int = 3) -> tf.keras.Model:
        """ Build a training model

        _extended_summary_

        Parameters
        ----------
        num_dense : int, optional
            the number of dense neural network layers, by default 3

        Returns
        -------
        tf.keras.Model
            a training model
        """

        # setup build specific parameters
        frac_dropout = config.ml.ML_frac_dropout
        activation_dense = config.ml.ML_activation_dense

        # Assemble the layers for the empty model
        # NOTE: Structure is:
        # =Text Input -> Preprocessor -> Encoder -> Dropout layer -> Dense layer

        # Build an empty ML model
        if self.verbose:
            logger.info("Building an empty ML model...")
            logger.info(f"Dropout fraction: {frac_dropout}")
            logger.info(f"Number of Dense layers: {num_dense}")

        # Text input layer
        layer_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")

        # Preprocessor layer
        ml_handle_preprocessor = self.mlconfig.dict_ml_model_preprocessors[self.model_key]
        ml_preprocessor = tfhub.KerasLayer(ml_handle_preprocessor)
        layer_preprocessor = tfhub.KerasLayer(ml_preprocessor, name="preprocessor")
        if self.verbose:
            logger.info(f"Loaded ML preprocessor: {ml_handle_preprocessor}")

        # Encoder layer
        ml_handle_encoder = self.mlconfig.dict_ml_model_encoders[self.model_key]
        ml_encoder = tfhub.KerasLayer(ml_handle_encoder, trainable=True)
        inputs_encoder = layer_preprocessor(layer_input)
        layer_encoder = tfhub.KerasLayer(ml_encoder, trainable=True, name="encoder")
        outputs_encoder = layer_encoder(inputs_encoder)
        if self.verbose:
            logger.info(f"Loaded ML encoder: {ml_handle_encoder}")

        # Dropout and Dense layers
        net = outputs_encoder["pooled_output"]
        net = tf.keras.layers.Dropout(frac_dropout)(net)
        net = tf.keras.layers.Dense(
            num_dense, activation=activation_dense, name="classifier"
        )(net)

        # Return the completed empty model
        return tf.keras.Model(layer_input, net)

    def split_datasets(self, train: str = None, test: str = None, validation: str = None):
        """_summary_

        _extended_summary_

        Parameters
        ----------
        train : str, optional
            _description_, by default None
        test : str, optional
            _description_, by default None
        validation : str, optional
            _description_, by default None
        """

        # load the data paths
        dir_train = train or os.path.join(self.data_dir, config.output.folders_TVT["train"])
        dir_test = test or os.path.join(self.data_dir, config.output.folders_TVT["test"])
        dir_validation = validation or os.path.join(self.data_dir, config.output.folders_TVT["validate"])

        # load the train, test, and validate datasets
        dataset_train_raw = tf.keras.preprocessing.text_dataset_from_directory(dir_train, batch_size=self.batch_size,
                                                                                label_mode=self.label_mode, seed=self.seed)
        dataset_test = tf.keras.preprocessing.text_dataset_from_directory(dir_test, batch_size=self.batch_size,
                                                                            label_mode=self.label_mode, seed=self.seed)
        dataset_validation = tf.keras.preprocessing.text_dataset_from_directory(dir_validation, batch_size=self.batch_size,
                                                                                label_mode=self.label_mode, seed=self.seed)

        # cache and prefetch the datasets
        self.dataset_train_raw = dataset_train_raw
        self.dataset_train = dataset_train_raw.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self.dataset_test = dataset_test.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self.dataset_validation = dataset_validation.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    def train_model(self, train: str = None, test: str = None, validation: str = None):
        """_summary_

        _extended_summary_

        Parameters
        ----------
        train : str, optional
            _description_, by default None
        test : str, optional
            _description_, by default None
        validation : str, optional
            _description_, by default None
        """

        # retrieve the split datasets
        self.split_datasets(train=train, test=test, validation=validation)

        # get the number of classes to use as layers in the model
        self.class_names = self.dataset_train_raw.class_names
        self.num_dense = len(self.class_names)

        # build the ML model
        self.model = self.build_model(num_dense=self.num_dense)
        if self.verbose:
            logger.info("Done building an empty ML model.")

        # Set up the loss, metric, and optimization functions
        if self.verbose:
            logger.info("Setting up loss, metric, and optimization functions...")

        init_loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = [tf.keras.metrics.CategoricalAccuracy("accuracy")]
        stepsize_epoch = tf.data.experimental.cardinality(self.dataset_train).numpy()

        # setup optimizer and fitting parameters
        frac_steps_warmup = config.ml.ML_frac_steps_warmup
        self.num_steps_train = stepsize_epoch * self.num_epochs
        self.num_steps_warmup = int(frac_steps_warmup * self.num_steps_train)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.init_lr)

        # Print some notes
        if self.verbose:
            logger.info(f"# of training steps: {self.num_steps_train}\n# of warmup steps: {self.num_steps_warmup}")
            logger.info(f"Type of optimizer and initial lr: {self.type_optimizer}, {self.init_lr}")

        # Compile the model with the loss, metric, and optimization functions
        self.model.compile(optimizer=optimizer, loss=init_loss, metrics=metrics)
        if self.verbose:
            logger.info("Done compiling loss, metric, and optimization functions.")
            logger.info(self.model.summary())

        # Set up callbacks
        log_dir = os.path.join(self.model_dir, 'logs')
        checkpoint_filepath = os.path.join(self.model_dir, 'checkpoints/checkpoint.weights.h5')

        # metric tracking dashboard
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # model save state, only save the weights
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True,
                                                                       monitor='loss', mode='min', save_freq='epoch',
                                                                       save_best_only=True, verbose=True)
        # hparams tracking
        self.hparams = {'num_dense_units': self.num_dense, 'batch_size': self.batch_size, 'num_epochs': self.num_epochs,
                        'learning_rate': self.init_lr}
        hparam_callback = hp.KerasCallback(log_dir, self.hparams)

        # early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=10,  # Number of epochs with no improvement after which training will be stopped
            restore_best_weights=True  # Restore model weights from the epoch with the best value
            )

        # Fit the model on the training data
        if self.verbose:
            logger.info("\nTraining the ML model...")

        self.history = self.model.fit(x=self.dataset_train, validation_data=self.dataset_validation, epochs=self.num_epochs,
                                 callbacks=[tensorboard_callback, model_checkpoint_callback, hparam_callback, early_stopping])

        # Set the loaded flag
        self.loaded = True

    def evaluate_model(self):
        """_summary_

        _extended_summary_

        Parameters
        ----------
        dataset_test : _type_
            _description_
        """
        # Evaluate the model on the test data
        if self.verbose:
            logger.info("\nTesting the ML model...")

        self.res_loss, self.res_accuracy = self.model.evaluate(self.dataset_test)


    def predict(self, texts: list):
        """_summary_

        _extended_summary_

        Parameters
        ----------
        texts : list
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # Run the model on the given texts
        results = self.model.predict(texts)

        # Print some notes
        if self.verbose:
            for ii, text in enumerate(texts):
                logger.info(f"{text}:\n{results[ii]}\n")

        return results

    def save_model(self):
        """_summary_

        _extended_summary_
        """
        # Save the model
        self.outputs = {
            "loss": self.res_loss,
            "class_names": self.class_names,
            "accuracy": self.res_accuracy,
            "init_lr": self.init_lr,
            "num_epochs": self.num_epochs,
            "num_steps_train": self.num_steps_train,
            "num_steps_warmup": self.num_steps_warmup,
            "type_optimizer": self.type_optimizer,
        }
        self.model.save(os.path.join(self.model_dir, self.savename_ML), include_optimizer=False)
        np.save(os.path.join(self.model_dir, self.savename_model), self.outputs)

    def load_model(self):
        """_summary_

        _extended_summary_
        """

        if not os.path.exists(self.filepath_ML):
            raise FileNotFoundError(f"Model output directory not found: {self.filepath_ML}.  Cannot load.")

        if not os.path.exists(self.filepath_model):
            raise FileNotFoundError(f"Numpy output file not found: {self.filepath_model}.  Cannot load.")

        # load the model
        optimizer = tf.keras.optimizers.Adam()
        self.model = tf.keras.models.load_model(
            self.filepath_ML, custom_objects={config.ml.ML_name_optimizer: optimizer}
        )

        # load the outputs
        self.outputs = np.load(self.filepath_model, allow_pickle=True).item()
        self.class_names = self.outputs["class_names"]

        # set the loaded flag
        self.loaded = True

    def plot_model(self):
        """_summary_

        _extended_summary_
        """
        # Extract variables
        num_epochs = self.outputs["num_epochs"]

        # For plot of loss and accuracy over time
        # For base plot
        e_arr = np.arange(0, num_epochs, 1)

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))
        fig.tight_layout()

        # For loss
        #ax = plt.subplot(2, 1, 1)
        ax1.set_title(f"Test loss: {self.outputs['loss']}\nTest accuracy: {self.outputs['accuracy']}")
        ax1.plot(e_arr, self.history.history["loss"], label="Loss: Training", color="blue", linewidth=4)
        ax1.plot(e_arr, self.history.history["val_loss"], label="Loss: Validation", color="gray", linewidth=2)
        leg = ax1.legend(loc="best", frameon=False)
        leg.set_alpha(0.5)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")

        # For accuracy
        #ax = plt.subplot(2, 1, 2)
        ax2.plot( e_arr, self.history.history["accuracy"], label="Accuracy: Training", color="blue", linewidth=4)
        ax2.plot( e_arr, self.history.history["val_accuracy"], label="Accuracy: Validation", color="gray", linewidth=2)
        leg = ax2.legend(loc="best", frameon=False)
        leg.set_alpha(0.5)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")

        # Save and close the plot
        fig.savefig(os.path.join(self.model_dir, "fig_model_lossandacc.png"))


def select_library(name: str) -> object:
    """ Select the model library class to use

    Parameters
    ----------
    name : str
        the name of the model library

    Returns
    -------
    object
        the model library class
    """
    for model in AbstractModel.__subclasses__():
        if model.__name__.lower() == name.lower():
            return model


class MLClassifier:

    def __init__(self, load: bool = False, verbose: bool = True):
        self.model = None
        self.verbose = verbose

        self.threshold = config.performance.threshold

        # Load the initial model from the specified library
        self.model_class = select_library(config.ml.ML_library)
        if not self.model_class:
            raise ValueError(f"Unsupported ML library: {config.ml.ML_library}")
        self.model = self.model_class(verbose=verbose, load=load)

    def run(self):
        # do nothing if model already loaded
        if self.model.loaded:
            return

        # train and evaluate the model
        self.model.train_model()
        self.model.evaluate_model()
        self.model.save_model()
        self.model.plot_model()

    def classify_text(self, text: str) -> dict:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        text : str
            _description_

        Returns
        -------
        dict
            _description_
        """
        # Print some notes
        if self.verbose:
            logger.info("\nRunning classify_text for ML classifier:")
            logger.info(f"Class names from model:\n{self.model.class_names}\n")

        # Clean the text
        # temporarily manually call Base until we can refactor
        # TODO - move out this code to keyword or grammer class
        base = Base()
        cleaned_text = base._streamline_phrase(text, do_streamline_etal=False)

        # Run model prediction on the text and map probabilities to classes
        probs = self.model.predict([cleaned_text])[0]
        prob_class_mapping = dict(zip(self.model.class_names, probs.astype('float64')))

        # Determine best verdict
        max_ind = np.argmax(probs)
        max_verdict = self.model.class_names[max_ind]

        # collect the results
        results = {"verdict": max_verdict, "scores_comb": None, "scores_indiv": None, "uncertainty": prob_class_mapping}

        # if result is below given threshold, return low-uncertainty verdict
        if (self.threshold is not None) and (probs[max_ind] < self.threshold):
            results = config.results.dictverdict_lowprob.copy()
            results["uncertainty"] = prob_class_mapping

        # Print some notes
        if self.verbose:
            logger.info("\nMethod classify_text for ML classifier complete!")
            logger.info(f"Max verdict: {max_verdict}\n")

        return results