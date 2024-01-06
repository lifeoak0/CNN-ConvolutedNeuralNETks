# CNN-ConvolutedNeuralNETks
[CNNsModelGEN_Predic.py] This code based on Python is responsible for the training of the Convolunted neural networks for the typical "two classification problem" in the case of the classification of cat and dog In this code several Python libs are used for training:
(1): Torch: A popular DL learning framework for building and training neural networks
(2：torch.NN：the neural network module in Pytorch provides various layers and functions required to build CNN.
(3):torch. optim: provide optimization algorithms such as Adam, for training the network
(4)torch-vision:  For processing image deta,including data transformation and loading pre-trained models>
(5)PIL (Python Imaging Library): used for image processing especially loading and transforming images in prediction funcs

====explanation for the KEY characteristics of the CNN in the [CNNsModelGEN_Predic.py]
1) convolutional layers（卷积层） : The model contains three convolutional layers (self.conv1, self.conv2, self.conv3), which are used to extract image features.
2) activation function (激活函数)： The model uses ReLU (Rectified Linear Unit) as the activation function, which is used after each convolutional layer and the first fully connected layer.
3）pooling layer: The maximum pooling layer (self. pool) is used in the model to reduce the spatial dimension of the feature map, reduce the amount of calculation and extract important features.
4) fully connected layers: The model contains two fully connected layers (self.fc1 and self.fc2) for decision-making based on features extracted by convolution and pooling layers.
5) SIGMOID激活函数：usually for the two classification problems.

=====Training and Predication process:
1)The model uses the Adam optimizer and the binary cross-entropy loss function (BCEWithLogitsLoss)
2)Data transformations include resizing images, converting to tensors, and normalizing, which help improve model performance.
3)When predicting, the PIL library is used to load and convert images, and then the model predicts them.

Explanation for the CS terminology in CNN and used characteristics:
1)  Adam optimizer: The adam optimizer is an optimization algorithm used in deep learning applications that commbines two extended stochastic gradient descent methods: momentum and adaptive learning rate (ADAgrad) the ADAM optimizer adjusts the learning rate by calculating the first order moment estimate the men and the second order moment of the uncertered variance of the gradient which makes it adaptive to the learning rate of different parameters , adjustment have been made to improve learning efficiency and stability>
2) convolution Layer: In computer science, especially in neural networks that process images, convolutional layers are used to extract features in images.This layer processes the image by applying a set of small filters (also called convolution kernels) that slide over the entire image to capture local features such as edges, color patches, or other texture information.In this way, the convolutional layer is able to learn meaningful image features from the raw pixel data.
3) activation function:The activation function is a key factor used to introduce nonlinearity in neural networks, which enables the network to learn and represent complex data such as images, sounds, etc.Applying an activation function to the output of each neural network layer can help the network capture nonlinear relationships in the data.Common activation functions include ReLU (rectified linear unit), which sets all negative values ​​to 0 while retaining positive values.
4) Pooling layers are another common layer used to reduce the number of parameters and computation in neural networks, while also reducing the risk of overfitting.Pooling operations typically downsample the output of a convolutional layer; for example, a max-pooling layer selects the maximum value within a region to represent that region.This not only reduces the dimensionality of the data, but also retains important feature information.
5) fully conneted layer: A fully connected layer is a layer in a neural network in which every input node is connected to every output node.In CNN, the fully connected layer is usually located at the end of the network and is used to make the final classification or regression decision based on the features extracted by the previous layers (convolutional layer and pooling layer).Their purpose is to synthesize the local features learned by previous layers to make the final prediction.
6) Sigmoid activation function:The Sigmoid activation function is a function that compresses the input value between 0 and 1. It is a smooth S-shaped curve.It is especially commonly used in binary classification problems because it can interpret the output as probability. For example, in the classification problem of cats and dogs, the output of sigmoid can be interpreted as "the probability of being a cat"
