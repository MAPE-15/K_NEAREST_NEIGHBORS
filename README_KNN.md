# K_NEAREST_NEIGHBORS  

### NOTES  
K-Nearest-Neighbors --> KNN  
One algorithm uses no sklearn and is made entirely from scratch.  
The other one uses the sklearn, and is the recommended one, because it's faster mainly.  

Also have breast-cancer-wisconsin.data dataset as an example, and to use this dataset with this algorithm.  

You pass k --> how many nearest points/neighbors from the feature to be predicted are taken.  
Then it takes k nearest points and the prediction is made by majority vote of the class, so if k=5 and got 3 Blues and 2 Greens, the prediction is --> Blue.  
Also you get the recommended k, and if don't know what k to use, remember k must be an odd number, and must not be a multiplier of the number of classes.  
