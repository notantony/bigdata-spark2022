package org.apache.spark.ml.made

import java.nio.file.Files

import breeze.linalg.sum
//import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.made.LinearRegressionTest.spark
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.util.random

import scala.util.Random

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val delta = 0.005
  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val vectors: Seq[Vector] = LinearRegressionTest._vectors
  lazy val coefficients: Seq[Double] = LinearRegressionTest._coefficients

  def generateLinearWithNoise2(c1: Double, c2: Double, intercept: Double, nExamples: Long, std: Double) : DataFrame = {
    val rdd1 = RandomRDDs.normalRDD(spark.sparkContext, nExamples).map(noise => noise * std + Random.nextDouble())
    val rdd2 = RandomRDDs.normalRDD(spark.sparkContext, nExamples).map(noise => noise * std + Random.nextDouble())
    val features : RDD[(Vector, Double)] = rdd1.zip(rdd2).map(
      t => (Vectors.dense(t._1, t._2), t._1 * c1 + t._2 * c2 + intercept)
    )

    val df = spark.createDataFrame(features)

    df.withColumnRenamed("_1", "features")
      .withColumnRenamed("_2", "label")
  }

  def generateLinearWithNoise3(c1: Double, c2: Double, c3: Double, intercept: Double, nExamples: Long, std: Double) : DataFrame = {
    val rdd1 = RandomRDDs.normalRDD(spark.sparkContext, nExamples).map(noise => noise * std + c1)
    val rdd2 = RandomRDDs.normalRDD(spark.sparkContext, nExamples).map(noise => noise * std + c2)
    val rdd3 = RandomRDDs.normalRDD(spark.sparkContext, nExamples).map(noise => noise * std + c3)
    val features : RDD[(Vector, Double)] = rdd1.zip(rdd2).zip(rdd3).map(
      t => (Vectors.dense(t._1._1, t._1._2, t._2), t._1._1 + t._1._2 + t._2 + intercept)
    )

    val df = spark.createDataFrame(features)

    df.withColumnRenamed("_1", "features")
      .withColumnRenamed("_2", "label")
  }

  "Model" should "approximate linear dependence" in {

    val model: LinearRegressionModel = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setOutputCol("prediction")
      .fit(data)

    val transformed = model.transform(data)
    val vectors: Array[Row] = transformed.collect()

    vectors.length should be(4)

    validatePredictions(model, model.transform(data), delta)
  }


  "Model" should "approximate linear dependence (y = x1 * 1.5 + x2 * 0.3 - 0.7)" in {
    val std = 0.01
    val coefficients = (-0.7, 1.5, 0.3)
    val generated = generateLinearWithNoise2(coefficients._2, coefficients._3, coefficients._1, 100000, std)

    generated.show(50)

    val model: LinearRegressionModel = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setOutputCol("prediction")
      .setLr(0.5)
      .setGradMomentum(0.9)
      .setMaxIterations(2000)
      .fit(generated)

    model.getWeights(0) should be (-0.7 +- delta)
    model.getWeights(1) should be (1.5 +- delta)
    model.getWeights(2) should be (0.3 +- delta)

    print(model.transform(generated).show())

    validatePredictions(model, model.transform(generated), delta = std)
  }


  private def validatePredictions(model: LinearRegressionModel, transformed: DataFrame, delta: Double): Unit = {
    transformed.columns should contain("prediction")
    transformed.columns should contain("label")

    transformed.collect()
      .map(row => (row.getAs[Double]("label"), row.getAs[Double]("prediction")))
      .foreach(p => (p._1 should be (p._2 +- delta)))
  }

  "Estimator" should "work after re-read" in {
    val sparkWakeUp = spark.sparkContext  // Fix SparkContext: Error initializing SparkContext, org.apache.spark.SparkException: A master URL must be set in your configuration
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setOutputCol("prediction")
    ))

    val tmpFolder = Files.createTempDirectory("scala-test1")
    pipeline.write.overwrite().save(tmpFolder.toAbsolutePath.toString)

    val reRead = Pipeline.load(tmpFolder.toAbsolutePath.toString)
    val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    validatePredictions(model, model.transform(data), delta)
  }

  "Model" should "work after re-read" in {
    val sparkWakeUp = spark.sparkContext  // Fix SparkContext: Error initializing SparkContext, org.apache.spark.SparkException: A master URL must be set in your configuration

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setOutputCol("prediction")
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDirectory("scala-test")
    model.write.overwrite().save(tmpFolder.toAbsolutePath.toString)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.toAbsolutePath.toString)

    validatePredictions(model.stages(0).asInstanceOf[LinearRegressionModel], reRead.transform(data), delta)
  }

  "Model" should "include bias/intercept term according to the config" in {

    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setOutputCol("prediction")
      .setMaxIterations(0)

    val model = estimator.fit(data)

    model.getWeights.length should be(data.columns.length + 1)

    estimator.setFitIntercept(false)

    val model2 = estimator.fit(data)

    model2.getWeights.length should be(data.columns.length)
  }
}

object LinearRegressionTest extends WithSpark {
  // y = 1.0 * x0 - 3.0 * x1 + 5.0
  lazy val _vectors = Seq(
    Vectors.dense(0.0, 0.0),
    Vectors.dense(1.0, 4.0),
    Vectors.dense(3.0, 2.0),
    Vectors.dense(-1.0, -2.0),
  )

  lazy val _labels = Seq(
    5.0,
    -6.0,
    2.0,
    10.0,
  )

  lazy val _coefficients = Seq(
    5.0,
    1.0,
    -3.0,
  )

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    _vectors.zip(_labels).map(p => (p._1, p._2)).toDF("features", "label")
  }
}
