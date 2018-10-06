# CS 7643 HW1 Q6.7 Extras
## Author
Zac Letian Chen
gt account: lchen483
EvalAI account name: Zac
EvalAI team name: lchen483

## Results
- Validation acc: $85%$
- Test acc (on local data): $85%$
- Test acc (on EvalAI): $87%$

## Architecture
I mainly tried VGG Net architecture. However, finally I didn't use any of the VGG architecture "VGG11", "VGG13", "VGG16" or "VGG19". I used a simplified version of "VGG13" with architecture as 

- Conv 64
- Conv 64
- Max pooling
- Conv 128
- Conv 128
- Max pooling
- Conv 256
- Conv 256
- Max pooling
- Fully connected 10

All convolution layer has kernel size of $3$, stride of $1$ and zero padding of $1$. All max pooling layer has kernel size of $2$ and stride of $2$. It is worth noting that after each convolution layer I add a Batch Normalization layer and then go through ReLU activation. 

The reason why I choose such an architecture is that it is relatively small compared with full VGG net (less convolution layer and less fully-connected layer). Considering my limited computation ability (one GTX 1080), a small network will let me be able to do more experiments. I also compared the performance of my model with "VGG13" and "VGG19". Given appropriate hyperparameter tuning, they behave almost the same, which means my model is powerful enough to handle CIFAR-10. 

## Other Modification
I modified `train.py` to let it support resume training. Specifically, I add a new command line parameter `--resume model_best.pth.tar`. If this parameter is added, the model will restore the model from `model_best.pth.tar`, restore the so-far best validation accuracy and epoch number, and then continue training based on that. The implementation is mainly test the model on validation set after each epoch, and if it does better than any previous model, save it to `model_best.pth.tar`. 

This function is super important, since we need to tweak hyperparameters a lot to get best performance, and we certainly don't want the model to be trained from zero every time. We could just resume the previously-best model and continue training on it. One most important hyperparameter to tune is the learning rate. We would like to use relatively large learning rate like $0.2$ at the beginning of the training as long as the loss is decreasing and validation score is increasing. After $0.2$ learning rate is too large to gain further performance, we could decrease it gradually to $0.05$, $0.01$, $0.001$. 

I also modify `cifar10.py` to let it support data augmentation. Specifically, I use two data augmentation methods in Pytorch: `transforms.RandomCrop()` and `transforms.RandomHorizontalFlip()`. When the `__getitem__` method is called and it is training phase, I would return randomly transformed image to help neural network training be more robust. 

## Notes
It is worth noting that figure "mymodel_lossvstrain.png" and "mymodel_valaccuracy.png" mean nothing because the "resume" training method I wrote above. Each re-training will overwrite the "mymodel.log" and the two figure rely on "mymodel.log" to draw. Therefore, "mymodel.log", "mymodel_lossvstrain.png" and "mymodel_valaccuracy.png" just reflect that last time I trained. I should have considered this and maintain "mymodel.log", but I have finished the train and it will take a lot of time to re-train it. Therefore, I hope TA could understand that and if you really want two figures I could provide them using some extra time. If that is the case, you could contact me via lchen483@gatech.edu. Thank you very much! 