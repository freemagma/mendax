# Machines are Among Us a.k.a mendax

Adversarial neural networks trained to 
identify each other in a simplified version of Among Us.

## Requirements

All requirements should be listed within `requirements.txt`. We reccomend utilizing
CUDA to speed up training time.

## Run a training session

First, edit `train.py` for your desired hyperparameters/input and output sizes.
Then, either call `train()` or `grid_search()` as desired. This should then start
the training session with your desired options and begin printing the output.
With gridsearch, ensure to edit the `grid_search()` method to perform your 
desired gridsearch. Finally, run `python 121q

## Plot a session or grid search

Uncomment the desired plot type method within `plot.py` and
change the parameters accordingly.