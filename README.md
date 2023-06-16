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
 - Best Test Accuracy: 99.05%
 - Best Train Accuracy: 99.50%

### Analysis
 - Number of parametrs increased a bit from 17360 to 17584
 - Gap in accuracy is reduced very much
 - Consistently accuracy is at 99.4+
 - Adding BatchNormalization helped to improve the training accuracy
 - Adding Dropout reduced a bit of overfitting and the gap beween train and test accuracies reduced


