# ERA-SESSION7

## Model0 : Preparing Structure
### Target:
 - Setup a proper structure

### Results:
 - No. of Params: 6,379,786
 - Best Test Accuracy: 99.32%
 - Best Train Accuracy: 99.96%

### Analysis:
 - Structure looks fine and model is learning
 - No. of params is too high, so needs to reduce
 - Accuracy looks fine for a basic model

## Model1 : Preparing a Skeleton
### Target
 - Setup a skeleton
 - Try to reduce the number of paramters

### Results
 - No. of Params: 75,024
 - Best Test Accuracy: 99.09%
 - Best Train Accuracy: 99.80%

### Analysis
 - Number of parametrs reduced from ~6 million to 75024
 - Accuracy is still maintained at 99+
 - Model architecure is more structured

## Model2 : Reduce No. of Params
### Target
 - Reduce the number of parameters
 - Maintain decent accuracy from last iteration of Model1
 - Use a transition block to reduce the channel count
 - Keep kernel numbers under 32 so as to reduce the number of paramters

### Results
 - No. of Params: 17,360
 - Best Test Accuracy: 99.05%
 - Best Train Accuracy: 99.50%

### Analysis
 - Number of parametrs reduced from ~75k to 17360
 - Accuracy is still maintained at 99+
 - There is a very little dip in both train and test accuracy
 - Reducing the number of kernels helped to reduce number of parameters
 - Adding a transition block allowed to reduce the kernel numbers thereby reducing the number of paramters

## Model3 : Improve Accuracy, Reduce Overfitting
### Target
 - Last model was showing scope for learning
 - So Trying to increase the training accuracy and maintain before attempting to reduce the number of parameters again
 - Use BatchNormalization to improve train and test accuracies
 - Add drop out so the gap between train and test accuracies reduce

### Results
 - No. of Params: 17,584
 - Best Test Accuracy: 99.42%
 - Best Train Accuracy: 99.53%

### Analysis
 - Number of parametrs increased a bit from 17360 to 17584
 - Gap in accuracy is reduced very much
 - Consistently accuracy is at 99.4+
 - Adding BatchNormalization helped to improve the training accuracy
 - Adding Dropout reduced a bit of overfitting and the gap beween train and test accuracies reduced

## Model4 : Reduce Number of Parameters
### Target
 - Reduce the number of parameters from 17k to 8k
 - Instead of going for kernel number from 16 > 32, approach is to change to 10 > 16
 - Add Gap layer to collapse image quickly while reducing the number of params

### Results
 - No. of Params: 8480
 - Best Test Accuracy: 99.34%
 - Best Train Accuracy: 99.06%

### Analysis
 - Number of parametrs reduced from 17584 to 8480
 - Gap layer helped to reduce the number of params by a great margin
 - Reducing the number of kernels helped too, still the params count is little higher than expected < 8k params
 - Accuracies have dipped a bit as a result of reduced number of params

## Model5 : Params < 8k & Acc. > 99.4%
### Target
 - Reduce the number of parameters to under 8k
 - Alter the kernel numbers from 10>16 to 10>10 and then 10>16, basically maintain kernel number and then increase, to reduce no of params
 - Add Rotational image transformations to increase the accuracies

### Results
 - No. of Params: 7550
 - Best Test Accuracy: 99.26%
 - Best Train Accuracy: 98.73%

### Analysis
 - Number of parametrs reduced from 8480 to 7550
 - Altering the layers approach helped to reduce the number of parameters
 - Even after image augmentaton maximum accuracy achieved is 99.26% :( :( :(



