**COMPUTER VISION PROJECT: TRANSFER LEARNING & FINE TUNING**

images folder download link: https://download.pytorch.org/tutorial/hymenoptera_data.zip 

**OVERVIEW**

This repo contains a deep neural network built with MobileNetV2 using transfer learning and finetuning techniques. 

**Library Dependency** 

This code has the following dependencies: 

    -Tensorflow

    -Numpy 

    -MatplotLib


**TRANSFER LEARNING APPROACHES**

1) **ConvNET for Fixed Feature Extraction:** 
The entire network iterates over 15 epochs. The first 10 epochs (1st - 10th epoch) freezes the entire weights of the imagenet 
base model(MobileNetV2) You may expect a similar result on compilation as the results shown below 


8/8 [==============================] - 5s 440ms/step - loss: 0.3275 - accuracy: 0.8531 - val_loss: 0.1739 - val_accuracy: 0.9477


Epoch 2/10
8/8 [==============================] - 3s 353ms/step - loss: 0.1252 - accuracy: 0.9551 - val_loss: 0.1295 - val_accuracy: 0.9477


Epoch 3/10
8/8 [==============================] - 3s 355ms/step - loss: 0.0950 - accuracy: 0.9592 - val_loss: 0.1331 - val_accuracy: 0.9412


Epoch 4/10
8/8 [==============================] - 3s 351ms/step - loss: 0.0526 - accuracy: 0.9837 - val_loss: 0.1743 - val_accuracy: 0.9412


Epoch 5/10
8/8 [==============================] - 3s 349ms/step - loss: 0.0665 - accuracy: 0.9796 - val_loss: 0.1449 - val_accuracy: 0.9542


Epoch 6/10
8/8 [==============================] - 3s 352ms/step - loss: 0.0430 - accuracy: 0.9796 - val_loss: 0.1479 - val_accuracy: 0.9542


Epoch 7/10
8/8 [==============================] - 3s 352ms/step - loss: 0.0585 - accuracy: 0.9673 - val_loss: 0.1536 - val_accuracy: 0.9542


Epoch 8/10
8/8 [==============================] - 3s 366ms/step - loss: 0.0710 - accuracy: 0.9633 - val_loss: 0.2250 - val_accuracy: 0.9412


Epoch 9/10
8/8 [==============================] - 3s 379ms/step - loss: 0.0272 - accuracy: 0.9918 - val_loss: 0.1805 - val_accuracy: 0.9477


Epoch 10/10
8/8 [==============================] - 3s 384ms/step - loss: 0.0173 - accuracy: 0.9959 - val_loss: 0.2103 - val_accuracy: 0.9542

154 

**2)Finetuning the ConvNet:** 

The last 5 steps of the algorithm (10th to 15th epoch) initialises with the pre-trained network 
to compute results. Expect a performance similar to this shown below on iteration: 


Epoch 10/15
8/8 [==============================] - 6s 471ms/step - loss: 0.0380 - accuracy: 0.9796 - val_loss: 0.2060 - val_accuracy: 0.9542


Epoch 11/15
8/8 [==============================] - 3s 398ms/step - loss: 0.0286 - accuracy: 0.9878 - val_loss: 0.2063 - val_accuracy: 0.9542


Epoch 12/15
8/8 [==============================] - 3s 393ms/step - loss: 0.0255 - accuracy: 0.9878 - val_loss: 0.2068 - val_accuracy: 0.9542


Epoch 13/15
8/8 [==============================] - 3s 395ms/step - loss: 0.0253 - accuracy: 0.9837 - val_loss: 0.2072 - val_accuracy: 0.9542


Epoch 14/15
8/8 [==============================] - 3s 394ms/step - loss: 0.0306 - accuracy: 0.9918 - val_loss: 0.2059 - val_accuracy: 0.9542


Epoch 15/15
8/8 [==============================] - 3s 390ms/step - loss: 0.0144 - accuracy: 1.0000 - val_loss: 0.2057 - val_accuracy: 0.9542


