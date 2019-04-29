# Credit Risk Prediction by Using Apache Spark MLlib

A random forest based classifier to predict the creditability of a person by using Apache Spark.

## Dataset  
- It is the [Statlog (German Credit Data) dataset](https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)) which classifies people described by a set of attributes as good or bad credit risks. 

## Approach
- Random forest classifier is an ensembled version of decision trees. It combines different decision trees in order to prevent overfitting.  
- In this project, I used Spark's scalable random forest classifier implementation from the MLlib library. More details can be found in the [documentation](https://spark.apache.org/docs/2.2.0/ml-classification-regression.html#random-forest-classifier).
- Apache Spark also has useful tools for hyperparameter tuning.[(*CrossValidator and TrainValidationSplit*)](https://spark.apache.org/docs/latest/ml-tuning.html) For this case, I used TrainValidationSplit which first splits the given data for training and validation, then trains and validates the model with different parameters in order to find the best one. It requires 4 different parameters as input:

1. **Parameter Grid** is used to construct a grid of parameters to search over the best random forest classifier model. For this case, 3 different values for maxBins and maxDepth, 2 different options for impurity are tried. (12 different models) 
```scala
val paramGrid = new ParamGridBuilder()
  .addGrid(RFclassifier.maxBins, Array(25,28, 31))
  .addGrid(RFclassifier.maxDepth, Array(4, 6, 8))
  .addGrid(RFclassifier.impurity, Array("entropy", "gini"))
  .build()  
```  
2. **Evaluator** metric is used to measure how well a fitted model handles the unseen test data.
```scala
val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
```  
3. **Pipeline** is the workflow representation of a sequence of algorithms to process and learn from data. 
```scala
val pipeline = new Pipeline().setStages(Array(RFclassifier.setSeed(1234)))
```  
4. **Training and Validation data** ratios have to be declared. 

By using these 4 components, TrainValidationSplit object is created in order to find the best fitted one of 12 models stated in parameter grid:  
```scala
val trainValidationSplit = new TrainValidationSplit()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setTrainRatio(0.75)
```

## Results
The best fitted model has an accuracy of 0.826. The hyperparameters of this model is given below:

```
Hyperparameters of the Best Model: 
{
	rfc_372ced9699eb-cacheNodeIds: false,
	rfc_372ced9699eb-checkpointInterval: 10,
	rfc_372ced9699eb-featureSubsetStrategy: auto,
	rfc_372ced9699eb-featuresCol: features,
	rfc_372ced9699eb-impurity: gini,
	rfc_372ced9699eb-labelCol: label,
	rfc_372ced9699eb-maxBins: 31,
	rfc_372ced9699eb-maxDepth: 6,
	rfc_372ced9699eb-maxMemoryInMB: 256,
	rfc_372ced9699eb-minInfoGain: 0.0,
	rfc_372ced9699eb-minInstancesPerNode: 1,
	rfc_372ced9699eb-numTrees: 20,
	rfc_372ced9699eb-predictionCol: prediction,
	rfc_372ced9699eb-probabilityCol: probability,
	rfc_372ced9699eb-rawPredictionCol: rawPrediction,
	rfc_372ced9699eb-seed: 1234,
	rfc_372ced9699eb-subsamplingRate: 1.0
}
```
A more detailed technical report can be found [here](./docs/report.pdf).
