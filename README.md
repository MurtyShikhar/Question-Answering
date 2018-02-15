# Match-LSTM and Answer Pointer (Wang and Jiang, ICLR 2016) #
This repo attempts to reproduce the match-lstm and answer pointer experiments from the 2016 paper on the same. A lot of the preprocessing boiler code is taken from Stanford CS224D. 

The meat of the code is in qa_model.py. I had to modify tensorflow's original attention mechanism implementation for the code to be correct. run train.py to train the model and qa_answer.py to generate answers given a set of paragraphs. Contact me at shikhar.murty@gmail.com for more info.

This code also serves as an example code showing how tensorflow's attention mechanism can be wired together. As of August 13th, 2017, such an example was not available anywhere.

## Preprocessing ##
Before training, you're going to want to do some preprocessing of the data.  Run the following from the command line:

```bash
$ python preprocessing/dwr.py
$ python preprocessing/squad_preprocess.py
$ python qa_data.py
```

The last step can take a bit of time (~30 minutes).

## Training ##
After preprocessing is complete, you can train your model by running the following command:

```bash
$ python train.py
```

Note that depending on your configs, this model will train for a very long time!  Given the default configs, on a modern laptop computer this will trian for multiple (~2) hours per epoch, and the default is 30 epochs (~60 hours).  