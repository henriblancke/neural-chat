## Neural Chatbot (seq2seq)

### Resources

1. [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)
2. [A Neural Conversational Model](https://arxiv.org/pdf/1506.05869.pdf)
3. [A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/pdf/1510.03055.pdf)
4. [A Persona-Based Neural Conversation Model](https://arxiv.org/pdf/1603.06155.pdf)

### Getting started

#### Install the requirements
```bash
pip install -r requirements.txt
```

#### Download train and test data
```bash
python s3.py --retrieve
```

#### Start training the model
```bash
python run.py
```

#### Test the trained model
```bash
python run.py --test

```

#### Save model checkpoints to S3
```bash
python s3.py --save
```

### Using beam search

A lot of the research building upon seq2seq to make an AI chatbot uses some approach of beam search. One of the aims of this project was to implement beam search. It is nonetheless still in an experimental phase.
The beam search approach in this project is based on the work of [Nikita Kitaev](https://gist.github.com/nikitakit/6ab61a73b86c50ad88d409bac3c3d09f).

To use beam search, either set the `beam_search` flag in the `variables.py` file to `True` or run the `run.py` script with `--beam_search=True` flag.

### Data input format

The input data is split into a training and testing dataset. Both the training and testing dataset contain Q&A or conversational turns.
The input file contains a question followed by its response or answer. For simplicity we forget the context of the conversation when looking
at conversational turns, there are ways to model conversational context.


### Parameters

#### Tuning the model parameters

Most of the models parameters can be customized in the `variables.py` file using TensorFlow flags.
All flag arguments are optional since reasonable default values are provided. More info below on tuning parameters.

#### Training parameters:

|Flag|Description|
|:---:|:---:|
|--learning_rate LEARNING_RATE                          |Learning rate.                         |
|--learning_rate_decay_factor LEARNING_RATE_DECAY_FACTOR|Learning rate decays by this much.     |
|--max_gradient_norm MAX_GRADIENT_NORM                  |Clip gradients to this norm.           |
|--steps_per_checkpoint STEPS_PER_CHECKPOINT      |How many training steps to do per checkpoint.|

#### Model architecture:

|Flag|Description|
|:---:|:---:|
|--batch_size BATCH_SIZE                                |Batch size to use during training.     |
|--size SIZE                                            |Size of each model layer.              |
|--num_layers NUM_LAYERS                                |Number of layers in the model.         |
|--vocab_size VOCAB_SIZE                                |Vocabulary size.                       |
|--model_type MODEL_TYPE               |Seq2Seq model type: 'embedding_attention' or 'embedding'|
|--buckets BUCKETS                                      |Implement the model with buckets       |
|--nobuckets                                            |
|--max_sentence_length  MAX_SENTENCE_LENGTH   |Maximum sentence length for model WITHOUT buckets|

#### Data parameters:

|Flag|Description|
|:---:|:---:|
|--max_train_data_size MAX_TRAIN_DATA_SIZE    |Limit on the size of training data (0: no limit).|
  
#### Directories:

|Flag|Description|
|:---:|:---:|
|--data_dir DATA_DIR                                    |Data directory.                        |
|--train_dir TRAIN_DIR                                  |Training directory.                    |
|--log_dir LOG_DIR                                      |Logging directory.                     |

### Roadmap
- Add Tensorboard 
- Add more advanced text pre-processing (see `preprocess.py`)
- Add option to use pre-trained embeddings
- Implement Diversity promotion