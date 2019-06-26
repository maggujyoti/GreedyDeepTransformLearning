# GreedyDeepTransformLearning
We introduce deep transform learning – a new tool for deep learning. Deeper representation is learnt by stacking one transform after another. The learning proceeds in a greedy way. The first layer learns the transform and features from the input training samples. Subsequent layers use the features (after activation) from the previous layers as training input. Experiments have been carried out with other deep representation learning tools – deep dictionary learning, stacked denoising autoencoder, deep belief network and PCA- Net (a version of convolutional neural network). Results show that our proposed technique is better than all the said techniques, at least on the benchmark datasets (MNIST, CIFAR-10 and SVHN) compared on.

Related Paper: J. Maggu and A. Majumdar, "Greedy deep transform learning," 2017 IEEE International Conference on Image Processing (ICIP), Beijing, 2017, pp. 1822-1826.

Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8296596&isnumber=8296222

Run file call_TL_deep.m for demo.
