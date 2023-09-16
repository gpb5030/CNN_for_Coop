This program implements two different CNN models on the MNIST and CIFAR10 
datasets. The program is implemented on pytorch. The CNN models achieves 
above 99% accuracy on MNIST and ~65% on CIFAR10 respectively.  

MINST CNN: The MINST dataset contains images that are 28x28 pixels.The first
convolution layer uses a 3x3 kernel with 10 out channels. Then a 2x2 max 
pooling. Then a 3x3 kernel on the 14x14 image with 20 out channels. Then 
another 2x2 max pooling. The channels are then flattened and fed through
a three layer full connected NN. The NN uses a 0.15 dropout rate which 
increased performance by ~0.50%. The model starts with a learning rate of
0.003 but decreases it slowly throughout the training process 
(learning rate * 0.985). The best performance achieved on this model was 
99.18% test accuracy.

CIFAR10: The CIFAR10 dataset contains colored (3 channel) 32x32 images. The
first CNN uses 3x3 kernel with 6 out channels. The next layer uses a 3x3 kernel
with 12 out channels. Then a 3x3 max pooling. Then a 3x3 convolution on the 14x
14 image. Then another 2x2 max pooling. 3 3x3 convolutions performed the
best out of any configuration tried. The channels are then flattened and 
fed through 3 fully connected layers. The dropout rate is 0.15 on all the
NN layers which helped to prevent overfitting. A lower learning rate of
0.002 was best for this model although it led to slower convergence. 
Similar to the MINST model, this model also has a decaying learning rate. The
best performance achieve for this model was 65 
