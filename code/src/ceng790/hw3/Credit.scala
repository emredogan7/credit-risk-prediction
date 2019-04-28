package ceng790.hw3

import org.apache.spark._
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.SQLContext

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions.rand


object Credit {
  // define the Credit Schema
  case class Credit(
    creditability: Double,
    balance: Double, duration: Double, history: Double, purpose: Double, amount: Double,
    savings: Double, employment: Double, instPercent: Double, sexMarried: Double, guarantors: Double,
    residenceDuration: Double, assets: Double, age: Double, concCredit: Double, apartment: Double,
    credits: Double, occupation: Double, dependents: Double, hasPhone: Double, foreign: Double)


  // function to create a  Credit class from an Array of Double
  def parseCredit(line: Array[Double]): Credit = {
    Credit(
      line(0),
      line(1) - 1, line(2), line(3), line(4), line(5),
      line(6) - 1, line(7) - 1, line(8), line(9) - 1, line(10) - 1,
      line(11) - 1, line(12) - 1, line(13), line(14) - 1, line(15) - 1,
      line(16) - 1, line(17) - 1, line(18) - 1, line(19) - 1, line(20) - 1)
  }


  // function to transform an RDD of Strings into an RDD of Double
  def parseRDD(rdd: RDD[String]): RDD[Array[Double]] = {
    rdd.map(_.split(",")).map(_.map(_.toDouble))
  }

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder.appName("Spark SQL").config("spark.master", "local[*]").getOrCreate()
    val sc = spark.sparkContext
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    import sqlContext.implicits._
    // load the data into a  RDD
    val creditDF = parseRDD(sc.textFile("credit.csv")).map(parseCredit).toDF().cache()
    println(creditDF.count())
    
    
    // shuffling the data before starting.
    val shuffledDF = creditDF.orderBy(rand())

    // question 1.
    // transforming a given list of columns (features) into a single vector column.
    val assembledFeatures = new VectorAssembler()
        .setInputCols(Array("balance", "duration", "history", "purpose", "amount","savings", "employment",
            "instPercent", "sexMarried", "guarantors", "residenceDuration", "assets", "age", "concCredit",
            "apartment","credits", "occupation", "dependents", "hasPhone", "foreign"))
        .setOutputCol("features")
        .transform(creditDF)

    assembledFeatures.select("features", "creditability").show(false)


    // question 2.
    //  encoding a string column of labels to a column of label indices.
    val indexer = new StringIndexer()
      .setInputCol("creditability")
      .setOutputCol("label")

    val indexedData = indexer
      .fit(assembledFeatures)
      .transform(assembledFeatures)

    indexedData.show

    // question 3.
    val splitWeights = Array(0.8, 0.2)
    val Array(train_data, test_data) = indexedData.randomSplit(splitWeights)

    // question 4.    
    val RFclassifier = new RandomForestClassifier()
      .setMaxDepth(6)
      .setMaxBins(28)
      .setImpurity("gini")
      .setFeatureSubsetStrategy("auto")
      .setSeed(1234)
    
    // the model creation without a pipeline; in other words, creating the random forest classifier model.
    val model = RFclassifier.fit(train_data)
    
    // printing the model description.
    println("\n\nMODEL SUMMARY: \n")
    println(model.toDebugString)

    // predictions of RFClassifier are taken and model accuracy is resulted.
    val modelPredictions = model.transform(test_data)
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")

    val modelAccuracy = evaluator.evaluate(modelPredictions)
    println("\nAccuracy of the model without using pipeline fitting :  " + modelAccuracy)
		
        
    // question 5.

    /* We use a ParamGridBuilder to construct a grid of parameters to search over
     * this grid will have 3 x 3 x 2 = 12 parameter settings for trainValidationSplit to choose from.
     * TrainValidationSplit will try all these combinations and choose best among them.
     */
    val paramGrid = new ParamGridBuilder()
      .addGrid(RFclassifier.maxBins, Array(25,28, 31))
      .addGrid(RFclassifier.maxDepth, Array(4, 6, 8))
      .addGrid(RFclassifier.impurity, Array("entropy", "gini"))
      .build()  

    // The pipeline consists of a single stage, our constructed random forest classifier.
    val pipeline = new Pipeline().setStages(Array(RFclassifier.setSeed(1234)))

    // Evaluator for binary classification, which expects two input columns: rawPrediction and label. 
    // val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
 
    /*  - TrainValidationSplit is used for hyperparameter tuning task, or model selection.
     *  - Estimator is the algorithm/pipeline to be tuned.
     *  - Evaluator is the metric to measure how well a fitted model does on held-out test data.
     *  - paramMaps corresponds to the changing parameters to find the best model.
     *  - trainValidationSplit is used for hyperparamter tuning. To be able to tune them, we cannot 
     *  use test data directly because we do not want our model to be fitted for specifically test data.
     *  Instead, we want a more generalized model. So, our test data should not be used for parameter tuning.
     *  A valdiation dataset is created (25% of train data = .8 * .25 = .2) from the 25% percent of original data.
     *   - In summary, 60% data => training
     *   						  20% data => validation
     *   							20% data => test
     */
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.75)

    /* By using 'TrainValidationSplit' for model selection, the best resulting model
     * is returned when trained with training data and validated by validation data. 
     * Then, we feed our test data to the model so that we can observe results on an unseen data!
     */
    val pipelineFittedModel = trainValidationSplit.fit(train_data)
    val pipelineFittedModelPredictions = pipelineFittedModel.transform(test_data)
    val pipelineAccuracy = evaluator.evaluate(pipelineFittedModelPredictions)
    
    println("Accuracy of the model with using pipeline fitting :  " + evaluator.evaluate(pipelineFittedModelPredictions))

    /* Among 12 different models, the best model is extracted so that
     * we can print its hyperparameters.
     */
    val bestModel = pipelineFittedModel.bestModel.
    asInstanceOf[org.apache.spark.ml.PipelineModel]
    .stages(0)

    // finding the parameters of best resulting method.
    val bestModelParameters = bestModel.extractParamMap()
    println("\n\nHyperparameters of the Best Model: \n")
    println(bestModelParameters)
        

  }
}
