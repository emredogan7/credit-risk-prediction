package edu.metu.ceng790.hw3

import org.apache.spark._
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.SQLContext

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._


object Credit {
  // define the Credit Schema
  case class Credit(
    creditability: Double,
    balance: Double, duration: Double, history: Double, purpose: Double, amount: Double,
    savings: Double, employment: Double, instPercent: Double, sexMarried: Double, guarantors: Double,
    residenceDuration: Double, assets: Double, age: Double, concCredit: Double, apartment: Double,
    credits: Double, occupation: Double, dependents: Double, hasPhone: Double, foreign: Double
  )
  // function to create a  Credit class from an Array of Double
  def parseCredit(line: Array[Double]): Credit = {
    Credit(
      line(0),
      line(1) - 1, line(2), line(3), line(4), line(5),
      line(6) - 1, line(7) - 1, line(8), line(9) - 1, line(10) - 1,
      line(11) - 1, line(12) - 1, line(13), line(14) - 1, line(15) - 1,
      line(16) - 1, line(17) - 1, line(18) - 1, line(19) - 1, line(20) - 1
    )
  }
  // function to transform an RDD of Strings into an RDD of Double
  def parseRDD(rdd: RDD[String]): RDD[Array[Double]] = {
    rdd.map(_.split(",")).map(_.map(_.toDouble))
  }

  def main(args: Array[String]) {

        val spark = SparkSession.builder.appName("Spark SQL").config("spark.master", "local[*]").getOrCreate()
				val sc = spark.sparkContext
				val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    import sqlContext.implicits._
    // load the data into a  RDD
    val creditDF = parseRDD(sc.textFile("credit.csv")).map(parseCredit).toDF().cache()
    println(creditDF.count())
   
  }
}

