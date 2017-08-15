# Match-LSTM and Answer Pointer (Wang and Jiang, ICLR 2016)
This repo attempts to reproduce the match-lstm and answer pointer experiments from the 2016 paper on the same. A lot of the preprocessing boiler code is taken from Stanford CS224D. 

The meat of the code is in qa_model.py. I had to modify tensorflow's original attention mechanism implementation for the code to be correct. run train.py to train the model and qa_answer.py to generate answers given a set of paragraphs. Contact me at shikhar.murty@gmail.com for more info.

This code also serves as an example code showing how tensorflow's attention mechanism can be wired together. As of August 13th, 2017, such an example was not available anywhere.
