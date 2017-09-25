# mir_torcs

## Preparation

### txt2csv.py

First of all, when we need to fix a tab separated value file in the Windows text file format, we can use ``txt2csv.py``. This will create a new csv file format in the Linux text file format.

``$ python txt2csv.py path_to_a_txt_file_name``

### config.py

This file has all necessary pieces of information for the training. You may change values when necessary. Note that this file does not need to be run by a user.

## Training

### train.py

After all parameters are properly set in ``config.py``, this ``train.py`` is the ultimate solution for the training. 

``$ python train.py path_to_a_folder_of_a_driving_data``

Note that you must use the path name of a driving data folder. This must not be a csv file name.
If you properly give a driving data folder, you will see followings.

```
Using TensorFlow backend.
100% (29200 of 29200) |###################| Elapsed Time: 0:00:11 Time: 0:00:11
Train samples:  20440
Valid samples:  8760
Epoch 1/5
 716/1277 [===============>..............] - ETA: 1373s - loss: 3.9270 - acc: 0.7354
```

### drive_data.py

This takes care of management of one drive data set which has captured images with steering angles and throttle values. Note that this file does not need to be run by a user.

### drive_train.py

This takes care of all training processes including CNN model building (we use ResNet by the way), preparation of training sets and validation sets, batch processes, and etc.

## Testing

Once a steering model is trained and created, it is time to test the model with another data set, which is often called a test data set.

### drive_test.py

This takes care of all details of the testing.

### test.py

To test a steering model with a test set, two arguments must be specified: a model name and a test set data folder name.

``$ python test.py steering_model_name test_data_folder_name``

the steering model name is a name of the weights (.h5) and the network model (.json) without an extension.

For example, if the weight fiel name is 2017-05-31-20-49-09.h5 and the network model name is 2017-05-31-20-49-09.json and they are at the folder ../mir_torcs_drive_data/, then you can use ``test.py`` as follows.

``$ python test.py ../mir_torcs_drive_data/2017-05-31-20-49-09 ../mir_torcs_drive_data/2017-05-31-17-26-11``

An example outpus is as follows.

```
Evaluating the model with test data sets ...
100% (930 of 930) |#######################| Elapsed Time: 0:06:41 Time: 0:06:41
Loss:  1.23780180987 Accuracy:  0.914424111948
```

