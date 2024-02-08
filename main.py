import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import os
import tensorflow.keras.layers as tfl
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation


#######################################################################################################
#CREATES TRAINING AND VALIDATION DATASET
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
train_directory = "hymenoptera_data/hymenoptera_data/train"
val_directory = "hymenoptera_data/hymenoptera_data/val"

#N.B: "Subset" and "validation_split" not called as image directory already splitted
train_dataset = image_dataset_from_directory(train_directory,
                                             shuffle=True,
                                             image_size=IMG_SIZE,
                                             #subset='training',
                                             #validation_split= 0.0
                                             )

validation_dataset = image_dataset_from_directory(val_directory,
                                             shuffle=True,
                                             image_size=IMG_SIZE,
                                             #subset='validation',
                                             #validation_split= 0.0
                                             )
###################################################################################################################
#PLOT SAMPLE IMAGES

# class_names = train_dataset.class_names
#
# plt.figure(figsize=(10, 10))
# for images, labels in train_dataset.take(1):
#     for i in range(9):
#         ax=plt.subplot(3,3, i + 1)
#         plt.imshow(images[i].numpy().astype("unit8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")

###################################################################################################################
#PREPROCESS AND AUGUMENT DATA

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

###################################################################################################################
#AUGUMENT DATA FUNCTION

def data_augumenter():
    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.2),
    ])
    return data_augmentation
#################################################################################################################
#MobileNet V2 Convolution Building Block

#STEP1: PreProcesses All Training Images
def preprocess (images, labels):
     #function processes image
    return tf.keras.applications.mobilenet_v2.preprocess_input(images), labels

train_dataset = train_dataset.map(preprocess)  #map over taining image directory
images, _ = next(iter(train_dataset.take(1)))
image = images[0]
#plt.imshow(image.numpy())
validation_dataset = validation_dataset.map(preprocess)

#STEP2: Collects MobileNetV2 Parameters' Summary
IMG_SHAPE = IMG_SIZE + (3, )
base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=IMG_SHAPE,
                                                            include_top=True,
                                                            weights='imagenet')
base_model.summary()

#STEP3: Gets last 2 layers of MobileNetV2 architecture (ie classification layers)
nb_layers = len(base_model.layers)
print(base_model.layers[nb_layers - 2].name)  #outputs: global_average_pooling2d
print(base_model.layers[nb_layers - 1].name)  #outputs: predictions


#STEP4: Iterates First Batch of Training Set on MobileNetV2
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape) # outputs: (32, 1000) i.e 32 batch size and 1000 classes
print(label_batch) #outputs: tf.Tensor([numpy_array], shape=(32,), dtype=int32)

#STEP5: Predictions From MobileNetV2 BaseModel
base_model.trainable = False
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
image_var= tf.Variable(preprocess_input(image_batch))
pred = base_model(image_var)

print(tf.keras.applications.mobilenet_v2.decode_predictions(pred.numpy(), top=2))

#############################################################################################################

#Layer_Freezing_With_The_Functional_API
def hymenoptera_model (image_shape=IMG_SIZE, data_augmentation=data_augumenter()):
    input_shape = image_shape + (3, )
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top= False,
                                                   weights = 'imagenet')

    #freeze the base model by making it non trainable
    base_model.trainable = False

    #create an input layer
    inputs = tf.keras.Input(shape=input_shape)

    #apply data augmentation to the input
    x = data_augumenter()(inputs)

    #set training to False to avoid keeping track of statistics in the batch
    x = base_model(x, training=False)

    #add the new Binary classification layers
    #use global avg pooling to summarise the info in each channel
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    #include dropout with probability of 0.2 to avoid overfitting
    x = tfl.Dropout(0.1)(x)
    x = tfl.Dense(64)(x)
    x = tfl.Dense(32)(x)
    x = tfl.Dense(32)(x)
    x = tfl.Dense(16)(x)
    x = tfl.Dropout(0.1)(x)

    #use a prediction layer with one neuron
    outputs = tfl.Dense(1)(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model

model2 = hymenoptera_model(IMG_SIZE, data_augmentation=data_augumenter())
print(model2.summary())

##############################################################################################################
#COMPILE MODEL
base_learning_rate = 0.0009
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
               metrics=['accuracy'])
initial_epochs = 10

history = model2.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)
#NB: result after 10th iteration = {loss: 0.0214 - accuracy: 0.9959 - val_loss: 0.1386 - val_accuracy: 0.9542}
##############################################################################################################

#FINE TUNING THE MODEL
#print(len(model2.layers))
base_model= model2.layers[2]
base_model.trainable= True
print(len(base_model.layers)) #outputs: 154 (ie number of layers in the model)

#Step1: Fine tune from this layers onwards
fine_time_from = 150

#Step2: Freeze All Layers Before It
for layer in base_model.layers[:fine_time_from]:
    layer.trainable = False

#Step3: Define Loss Function CrossEntropy, optimizer, metrics
loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate= 0.003/ 900)
metrics = ["accuracy"]

#Step4: Compile and Fit New Finetune Model
model2.compile(loss = loss_function,
               optimizer= optimizer,
               metrics=metrics)

fine_tune_epochs = 5
total_epochs = initial_epochs + fine_tune_epochs
finetune_history = model2.fit(train_dataset,
                              epochs=total_epochs,
                              initial_epoch=history.epoch[-1],
                              validation_data=validation_dataset)
