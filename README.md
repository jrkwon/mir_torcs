# mir_torcs (v0.4)

## Preparation

### txt2csv.py

First of all, when we need to fix a tab separated value file in the Windows text file format, we can use ``txt2csv.py``. This will create a new csv file format in the Linux text file format.

``$ python txt2csv.py path_to_a_txt_file_name``

### config.py

This file has all necessary pieces of information for the training. You may change values when necessary. Note that this file does not need to be run by a user.

## Tools
### drive_data.py
This takes care of management of one drive data set which has captured image names with steering angles and throttle values. Note that this file does not need to be run by a user.

### image_process.py
This class is for image processing. For now, it has only one function for the histogram equalization.

## Network model
### net_model
This class is for the CNN model. In this particular case where we are training steering angles which are floating point numbers, it is not a good idea to use 'accuracy' as a metric. Also, the activation function for the output must be one of linear functions. Do not use something like softmax which generates values from 0 to 1 since the steering angle values are between -1 and 1.

## Training

The training was not very successful in the early stages. So I did some changes in the training after investigating the training data set.

### Scaling steering angles
Most steering angle values are between -0.2 and 0.2. Only few pick values are found. To make the value be in the range -1 and 1, I used 5 as a multiplication factor to the steering angle. And when a trained model's prediction must scale down by the factor of 5.  

### Jitters
Since we collected data from a human driver's behavior, the steering angle has small jitters that prevent a network from being properly trained. So I removed those jitters. After scrutinizing the data sets, 0.002 was chosen as the minimum steering angle value. This means that if a steering angle is smaller than 0.002, the value will be considered as 0. This is done both in the training and prediction stage.

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
 716/1277 [===============>..............] - ETA: 1373s - loss: 3.9270 
```


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
100% (14866 of 14866) |###################| Elapsed Time: 0:00:05 Time: 0:00:05
  1% (3 of 233) |                          | Elapsed Time: 0:00:00 ETA: 0:00:09
  Test samples:  14866

Evaluating the model with test data sets ...
100% (233 of 233) |#######################| Elapsed Time: 0:00:22 Time: 0:00:22
  3% (8 of 233) |                          | Elapsed Time: 0:00:00 ETA: 0:00:22
Loss:  0.00106151084765
```

## Validating a trained model
### drive_batch.py
This class manages a batch test of a trained model. 

### run_batch.py
This is an example of how to use the `DriveBatch` class in the `drive_batch.py`

```
Test samples:  14866
100% (14866 of 14866) |###################| Elapsed Time: 0:01:19 Time: 0:01:19
```

After executing this script, you will see a csv file in the data folder that you specified to test. 

```
image_name, label, predict, abs_error
2017-05-31-17-26-17-249000,0.0,0.00315131,0.00315130874515
2017-05-31-17-26-17-348000,0.00784301757812,-0.000153585,0.00799660210032
2017-05-31-17-26-17-481000,0.00784301757812,0.0029467,0.00489631621167
2017-05-31-17-26-17-581000,0.00784301757812,0.00366228,0.00418073590844
2017-05-31-17-26-17-698000,0.00784301757812,0.00693941,0.000903607346112
2017-05-31-17-26-17-798000,0.00784301757812,0.00306049,0.00478252582252
2017-05-31-17-26-17-864000,0.00784301757812,0.00173601,0.00610700680408
2017-05-31-17-26-17-981000,0.00784301757812,0.00432915,0.00351387076079
2017-05-31-17-26-18-082000,0.00784301757812,0.00429135,0.00355166289955
2017-05-31-17-26-18-197000,0.00784301757812,0.00110958,0.0067334343912
2017-05-31-17-26-18-315000,0.00784301757812,0.00784485,1.83191150927e-06
2017-05-31-17-26-18-430000,0.00784301757812,-0.00630262,0.0141456346028
2017-05-31-17-26-18-548000,0.00784301757812,0.00188739,0.00595562497619
2017-05-31-17-26-18-649000,0.00784301757812,-0.00523483,0.0130778523162
2017-05-31-17-26-18-781000,0.00784301757812,-0.00412897,0.0119719831273
2017-05-31-17-26-18-882000,0.00784301757812,0.00614799,0.0016950299032
2017-05-31-17-26-18-999000,0.00784301757812,0.00537084,0.00247217807918
2017-05-31-17-26-19-097000,0.00784301757812,0.0024988,0.00534421578049
2017-05-31-17-26-19-214000,0.00784301757812,0.000409422,0.00743359571788
2017-05-31-17-26-19-331000,0.00784301757812,-0.000464639,0.00830765685532
2017-05-31-17-26-19-447000,0.00784301757812,-0.000580866,0.00842388358432
2017-05-31-17-26-19-547000,0.00784301757812,0.0022979,0.00554511602967
...
```

You can check the `abs_error` to see the trained model's accuracy in general.


## Example of using a trained model
### drive_run.py
This is a simple class to show how to run the trained model.

### run.py 
If you give a trained model name and an image as input, this will load the model and print out the prediction value. 

## TORCS driver using a trained model
### snakeoil.py
A modified version of snakeoil.py originally written by Chris X Edwards.

### drive_torcs.py
```
$python drive_torcs.py steering_model_name
```
