package ceng790.hw3

import org.apache.spark._
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.SQLContext

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler

import org.apache.spark.ml.tuning.{ ParamGridBuilder, TrainValidationSplit}

import org.apache.spark.ml.{ Pipeline, PipelineStage }
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.log4j.{ Level, Logger }

object Credit {
  // define the Credit Schema
  case class Credit(
    creditability: Double,
    balance:       Double, duration: Double, history: Double, purpose: Double, amount: Double,
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

    // question 1.
    val assembledFeatures = new VectorAssembler()
        .setInputCols(Array("balance", "duration", "history", "purpose", "amount","savings", "employment",
            "instPercent", "sexMarried", "guarantors", "residenceDuration", "assets", "age", "concCredit",
            "apartment","credits", "occupation", "dependents", "hasPhone", "foreign"))
        .setOutputCol("features")
        .transform(creditDF)

    // question 2.
    val indexer = new StringIndexer()
      .setInputCol("creditability")
      .setOutputCol("label")

    val indexedData = indexer
      .fit(assembledFeatures)
      .transform(assembledFeatures)

    indexedData.show

    // question 3.
    val splitWeights = Array(0.8, 0.2)
    // val splitRandomnessSeed = 1234
    val Array(train_data, test_data) = indexedData.randomSplit(splitWeights)

    // question 4.
    val RFclassifier = new RandomForestClassifier()
    .setMaxDepth(3)
    .setMaxBins(25)
    .setImpurity("gini")
    .setFeatureSubsetStrategy("auto")
    .setSeed(1234)

    // the model creation without a pipeline
    val model = RFclassifier.fit(train_data)
    println("\n\nMODEL SUMMARY: \n")
    println(model.toDebugString)

    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
    val predictions = model.transform(test_data)

    val accuracy = evaluator.evaluate(predictions)
    println("\nAccuracy of the model without using pipeline fitting :  " + accuracy)

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

    // The pipeline consists of a single stage, our constructed classifier.
    val pipeline = new Pipeline().setStages(Array(RFclassifier))

    /* TrainValidationSplit is used for hyperparameter tuning task, or model selection.
     * Estimator is the algorithm/pipeline to be tuned.
     * Evaluator is the metric to measure how well a fitted model does on held-out test data.
     */
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.75)

    /* By using 'TrainValidationSplit' for model selection, the best resulting model
     * is achieved when trained with training data. Notice that 75% of training data
     * is used for actual training stage and the rest for the hyperparameter selection (validation)!
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
    println(bestModelParameters)



  }
}
