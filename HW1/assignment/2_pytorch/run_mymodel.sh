#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model mymodel \
    --epochs 20 \
    --weight-decay 0.0005 \
    --momentum 0.9 \
    --batch-size 128 \
    --lr 0.1 | tee mymodel.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
