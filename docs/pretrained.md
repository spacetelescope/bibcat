# Using Pretrained Models (BERT)

Here we describe the `bibcat` options for classifying papers using pretrain models like BERT or RoBERTa.

## Quick Start
Here is a quick start guide. If this is your first time running `bibcat`, you will need to create and build a training model first, then classify papers using the trained model.

### Train the ML models

1. To create a training model, configure the name_model parameter within the output: section of the bibcat_config.yaml file. By default, this parameter is set to tf_bert_run, but you can customize it to any name that aligns with your project or run identifier.
  ```
  output:
    name_model: tf_bert_run
  ```
2. run `bibcat ml train`. This process will build and train a new model using the default machine learning (ML) hyperparameter settings specified in the ml: section of the bibcat_config.yaml file, applying them to the training dataset.

### Evaluate the trained models
- To evaluate the classifiers, run `bibcat ml evaluate`. `fetch_papers.py` (with `do_evaluation=True`) fetches the test papers with papertrack classification (`paper_type`). It will produce some evaluation diagnostics such as a confusion matrix in the `output/ouptut/` directory.

### Paper classification for operation
- To classify papers with the trained model, run `bibcat ml classify`. Copy `etc/fakedata.json` to your local OPSDATA folder to test `bibcat ml classify`. Check out `etc/fakedata.json` to see the necessary contents for operational papers in JSON. You use any texts with their bibcode in a JSON file by pointing `inputs.path_ops_data`in `bibcat_config.yaml` to your JSON file. fetch_papers.py (with with `do_evaluation=False` will fetch the JSON file for classification )


### Changing Models

`bibcat` now supports the ability to use other Tensorflow models for paper classification. The default model used is `bert`.  New models are added into bibcat via the bibcat [configuration yaml](https://github.com/spacetelescope/bibcat/blob/dev/bibcat/etc/bibcat_config.yaml) file, under the `ml` section, similar to the existing `bert` section.  Then, update the `ML_model_type` and `ML_model_key` keys to the new model values.

For example to use the `roberta` model, with roberta-specific encoders/preprocessors, within your user `$BIBCAT_CONFIG_DIR/bibcat_config.yaml`, you would set:
```yaml
output:
  name_model: tf_roberta_trial
ml:
  ML_model_type: "roberta"
  ML_model_key: "roberta_encased"
  roberta:
    dict_ml_model_encoders: {"roberta_encased": "https://www.kaggle.com/models/kaggle/roberta/TensorFlow2/en-cased-l-12-h-768-a-12/1"}
    dict_ml_model_preprocessors: {"roberta_encased": "https://kaggle.com/models/kaggle/roberta/TensorFlow2/en-cased-preprocess/1"}
```
Then run `bibcat ml train` and `bibcat ml classify` as normal.

Alternatively, you can specify new models directly from the command line during `bibcat ml train`.  For example, to use the `roberta` model, run:
```
bibcat ml train -m roberta -n tf_roberta_trial -k roberta_encased
```
This assumes the preprocessors and encoders for that model are already included in the config file.  To use a different preprocessor or encoder not included in your configuration file, you can manually pass in the urls, e.g.
```
bibcat ml train -m roberta -n tf_roberta_trial -k roberta_encased -e https://www.kaggle.com/models/kaggle/roberta/TensorFlow2/en-cased-l-12-h-768-a-12/1 -p https://kaggle.com/models/kaggle/roberta/TensorFlow2/en-cased-preprocess/1
```
