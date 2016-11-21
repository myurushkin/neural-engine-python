# Simple neural network engine in python

Implementation of stochastic gradient descent for neural networks. The following layers are implemented:
 * Dense.
 * Dropout.
 * Sigmoid.
 * Softmax.
The last layer has to be softmax with cross-entropy loss function.

Implemented model was trained on MNIST dataset and it outperformed keras+tensorflow model on the MNIST dataset.
It's very strange, but I didn't find an error in my code at the moment.

## My model train/test results:
```
epoch: 0, train_acc=0.49614094705, train_loss=1.41596842708, test_accuracy=0.8375
epoch: 1, train_acc=0.844954912935, train_loss=0.50296318476, test_accuracy=0.8908
epoch: 2, train_acc=0.890680525942, train_loss=0.364993919197, test_accuracy=0.9188
epoch: 3, train_acc=0.90631663113, train_loss=0.308670257308, test_accuracy=0.9264
epoch: 4, train_acc=0.918504575338, train_loss=0.264555058292, test_accuracy=0.9383
epoch: 5, train_acc=0.929443185856, train_loss=0.230611740587, test_accuracy=0.9451
epoch: 6, train_acc=0.935662091329, train_loss=0.210300688771, test_accuracy=0.9489
epoch: 7, train_acc=0.940354033404, train_loss=0.191236859757, test_accuracy=0.9474
epoch: 8, train_acc=0.945978811301, train_loss=0.176755872023, test_accuracy=0.9569
epoch: 9, train_acc=0.949715707178, train_loss=0.163037055785, test_accuracy=0.9575
epoch: 10, train_acc=0.952081112296, train_loss=0.155110967763, test_accuracy=0.9626
epoch: 11, train_acc=0.954674173774, train_loss=0.143791986874, test_accuracy=0.9644
epoch: 12, train_acc=0.956928527008, train_loss=0.135548958311, test_accuracy=0.9661
epoch: 13, train_acc=0.961742626155, train_loss=0.123447152679, test_accuracy=0.9672
epoch: 14, train_acc=0.96210909737, train_loss=0.120634633647, test_accuracy=0.9709
epoch: 15, train_acc=0.962581067875, train_loss=0.117321749307, test_accuracy=0.9672
epoch: 16, train_acc=0.965296286425, train_loss=0.110093101205, test_accuracy=0.9725
epoch: 17, train_acc=0.96650119936, train_loss=0.106084837919, test_accuracy=0.9728
epoch: 18, train_acc=0.96914978678, train_loss=0.0995051852615, test_accuracy=0.9734
epoch: 19, train_acc=0.969893834399, train_loss=0.0959316519439, test_accuracy=0.9735
accuracy of my model: 0.9735
```
## keras+tensorflow train/test results
```
Train on 60000 samples, validate on 10000 samples
Epoch 1/40
60000/60000 [==============================] - 8s - loss: 2.2651 - acc: 0.1625 - val_loss: 1.9281 - val_acc: 0.4522
Epoch 2/40
60000/60000 [==============================] - 8s - loss: 1.3979 - acc: 0.5074 - val_loss: 0.8987 - val_acc: 0.6940
Epoch 3/40
60000/60000 [==============================] - 8s - loss: 0.8446 - acc: 0.7150 - val_loss: 0.6033 - val_acc: 0.8170
Epoch 4/40
60000/60000 [==============================] - 8s - loss: 0.6439 - acc: 0.7911 - val_loss: 0.4769 - val_acc: 0.8599
Epoch 5/40
60000/60000 [==============================] - 8s - loss: 0.5526 - acc: 0.8257 - val_loss: 0.4330 - val_acc: 0.8720
Epoch 6/40
60000/60000 [==============================] - 8s - loss: 0.5009 - acc: 0.8445 - val_loss: 0.3843 - val_acc: 0.8871
Epoch 7/40
60000/60000 [==============================] - 8s - loss: 0.4678 - acc: 0.8564 - val_loss: 0.3641 - val_acc: 0.8942
Epoch 8/40
60000/60000 [==============================] - 8s - loss: 0.4426 - acc: 0.8654 - val_loss: 0.3493 - val_acc: 0.8983
Epoch 9/40
60000/60000 [==============================] - 8s - loss: 0.4256 - acc: 0.8717 - val_loss: 0.3374 - val_acc: 0.9009
Epoch 10/40
60000/60000 [==============================] - 9s - loss: 0.4085 - acc: 0.8775 - val_loss: 0.3266 - val_acc: 0.9033
Epoch 11/40
60000/60000 [==============================] - 9s - loss: 0.3942 - acc: 0.8813 - val_loss: 0.3167 - val_acc: 0.9065
Epoch 12/40
60000/60000 [==============================] - 10s - loss: 0.3866 - acc: 0.8841 - val_loss: 0.3091 - val_acc: 0.9098
Epoch 13/40
60000/60000 [==============================] - 8s - loss: 0.3713 - acc: 0.8882 - val_loss: 0.3038 - val_acc: 0.9102
Epoch 14/40
60000/60000 [==============================] - 8s - loss: 0.3647 - acc: 0.8912 - val_loss: 0.2924 - val_acc: 0.9135
Epoch 15/40
60000/60000 [==============================] - 8s - loss: 0.3548 - acc: 0.8951 - val_loss: 0.2849 - val_acc: 0.9164
Epoch 16/40
60000/60000 [==============================] - 8s - loss: 0.3459 - acc: 0.8965 - val_loss: 0.2805 - val_acc: 0.9179
Epoch 17/40
60000/60000 [==============================] - 8s - loss: 0.3385 - acc: 0.8991 - val_loss: 0.2733 - val_acc: 0.9195
Epoch 18/40
60000/60000 [==============================] - 6s - loss: 0.3299 - acc: 0.9019 - val_loss: 0.2670 - val_acc: 0.9205
Epoch 19/40
60000/60000 [==============================] - 7s - loss: 0.3221 - acc: 0.9035 - val_loss: 0.2627 - val_acc: 0.9220
Epoch 20/40
60000/60000 [==============================] - 7s - loss: 0.3154 - acc: 0.9051 - val_loss: 0.2565 - val_acc: 0.9247
Epoch 21/40
60000/60000 [==============================] - 6s - loss: 0.3072 - acc: 0.9071 - val_loss: 0.2507 - val_acc: 0.9251
Epoch 22/40
60000/60000 [==============================] - 6s - loss: 0.3001 - acc: 0.9096 - val_loss: 0.2440 - val_acc: 0.9279
Epoch 23/40
60000/60000 [==============================] - 7s - loss: 0.2958 - acc: 0.9107 - val_loss: 0.2403 - val_acc: 0.9285
Epoch 24/40
60000/60000 [==============================] - 8s - loss: 0.2891 - acc: 0.9132 - val_loss: 0.2349 - val_acc: 0.9308
Epoch 25/40
60000/60000 [==============================] - 7s - loss: 0.2832 - acc: 0.9157 - val_loss: 0.2334 - val_acc: 0.9303
Epoch 26/40
60000/60000 [==============================] - 7s - loss: 0.2769 - acc: 0.9167 - val_loss: 0.2286 - val_acc: 0.9333
Epoch 27/40
60000/60000 [==============================] - 7s - loss: 0.2737 - acc: 0.9170 - val_loss: 0.2208 - val_acc: 0.9330
Epoch 28/40
60000/60000 [==============================] - 7s - loss: 0.2699 - acc: 0.9175 - val_loss: 0.2185 - val_acc: 0.9341
Epoch 29/40
60000/60000 [==============================] - 6s - loss: 0.2628 - acc: 0.9207 - val_loss: 0.2140 - val_acc: 0.9355
Epoch 30/40
60000/60000 [==============================] - 7s - loss: 0.2600 - acc: 0.9215 - val_loss: 0.2107 - val_acc: 0.9361
Epoch 31/40
60000/60000 [==============================] - 8s - loss: 0.2544 - acc: 0.9233 - val_loss: 0.2080 - val_acc: 0.9379
Epoch 32/40
60000/60000 [==============================] - 6s - loss: 0.2510 - acc: 0.9241 - val_loss: 0.2044 - val_acc: 0.9384
Epoch 33/40
60000/60000 [==============================] - 6s - loss: 0.2469 - acc: 0.9252 - val_loss: 0.1998 - val_acc: 0.9399
Epoch 34/40
60000/60000 [==============================] - 7s - loss: 0.2408 - acc: 0.9266 - val_loss: 0.1981 - val_acc: 0.9399
Epoch 35/40
60000/60000 [==============================] - 6s - loss: 0.2399 - acc: 0.9269 - val_loss: 0.1939 - val_acc: 0.9416
Epoch 36/40
60000/60000 [==============================] - 7s - loss: 0.2336 - acc: 0.9293 - val_loss: 0.1911 - val_acc: 0.9414
Epoch 37/40
60000/60000 [==============================] - 8s - loss: 0.2321 - acc: 0.9299 - val_loss: 0.1878 - val_acc: 0.9430
Epoch 38/40
60000/60000 [==============================] - 8s - loss: 0.2313 - acc: 0.9306 - val_loss: 0.1872 - val_acc: 0.9434
Epoch 39/40
60000/60000 [==============================] - 9s - loss: 0.2244 - acc: 0.9319 - val_loss: 0.1828 - val_acc: 0.9447
Epoch 40/40
60000/60000 [==============================] - 8s - loss: 0.2206 - acc: 0.9327 - val_loss: 0.1797 - val_acc: 0.9471
```
