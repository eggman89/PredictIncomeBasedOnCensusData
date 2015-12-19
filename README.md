# Predict Income Based On CensusData
#### Using MLLib
Predict whether income exceeds $50K/yr based on census data.

Dataset: [UCI page](https://archive.ics.uci.edu/ml/datasets/Adult) [Direct Link](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/)

This project train [gradient boosting](https://spark.apache.org/docs/1.2.1/mllib-ensembles.html#Gradient-Boosted-Trees-(GBTS)) with around 30k enteries of adult persons. Our aim is to correctly predict if the the users from the test data would would earn more than 50k dollar a year or not.

-Result: We run the test data as it is on the model. The accuracy is about 92%. 


