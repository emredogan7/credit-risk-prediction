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
    val labeledData = new StringIndexer()
    .setInputCol("creditability")
    .setOutputCol("label")
    .fit(assembledFeatures)
    .transform(assembledFeatures)
    labeledData.show

    // question 3. 
    
    val splitWeights = Array(0.8,0.2)
    val splitRandomnessSeed = 1234
    val Array(train_data, test_data) = labeledData.randomSplit(splitWeights, splitRandomnessSeed)

    
    // question 4.
    val classifier = new RandomForestClassifier()
    .setMaxDepth(3)
    .setMaxBins(25)
    .setImpurity("gini")
    .setFeatureSubsetStrategy("auto")
    .setSeed(1234)
    
    // the model creation without a pipeline
    val model = classifier.fit(train_data)

    println(model.toDebugString)

    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
    val predictions = model.transform(test_data)

    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy of the model without using pipeline fitting :  " + accuracy)


    
    // question 5. 

    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.maxBins, Array(25,28, 31))
      .addGrid(classifier.maxDepth, Array(4, 6, 8))
      .addGrid(classifier.impurity, Array("entropy", "gini"))
      .build()

    val steps: Array[PipelineStage] = Array(classifier)
    val pipeline = new Pipeline().setStages(steps)
    
    
    val trainValidationSplit = new TrainValidationSplit()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setTrainRatio(0.75)

    // model creation by using a pipeline.
    
    val pipelineFittedModel = trainValidationSplit.fit(train_data)

    val pipelinePredictions = pipelineFittedModel.transform(test_data)
    
    pipelinePredictions
      .select("features", "label", "prediction")
      .show(100)
    
    val pipelineAccuracy = evaluator.evaluate(pipelinePredictions)
    println("Accuracy of the model with using pipeline fitting :  " + pipelineAccuracy)

    val bestModel = pipelineFittedModel.bestModel.
    asInstanceOf[org.apache.spark.ml.PipelineModel]
    .stages(0)
    
    val bestModelParameters = bestModel.extractParamMap()
    println(bestModelParameters)



  }
}
