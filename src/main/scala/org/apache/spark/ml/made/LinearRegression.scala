package org.apache.spark.ml.made

import breeze.linalg.{*, max}
import breeze.numerics.abs
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{BooleanParam, DoubleArrayParam, DoubleParam, IntParam, LongParam, Param, ParamMap}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasFitIntercept, HasInputCol, HasInputCols, HasLabelCol, HasMaxIter, HasOutputCol, HasOutputCols}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.{ml, mllib}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.vectorized.ColumnVector
import org.apache.spark.sql.{Column, DataFrame, Dataset, Encoder, Row, functions}

import scala.util.Random

trait LinearRegressionParams
  extends HasFeaturesCol
    with HasLabelCol
    with HasOutputCol
    with HasFitIntercept
    with HasMaxIter
{
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)
  def setMaxIterations(value: Int): this.type = set(maxIter, value)

  val lr = new DoubleParam(this, "lr", "Learning rate")
  def getLr: Double = $(lr)
  def setLr(value: Double): this.type = set(lr, value)

  val minGradDelta = new DoubleParam(this, "minGradDelta",
    "Early stopping criterion, stop training process if the weight change on every axis is smaller than specified value")
  def getMinGradDelta: Double = $(minGradDelta)
  def setMinGradDelta(value: Double): this.type = set(minGradDelta, value)

  // Assume model is small enough to fit into memory
  val weights = new DoubleArrayParam(this, "weights", "Linear regression feature coefficients")
  // Empty until the training is performed
  def getWeights: Array[Double] = $(weights)
  def setWeights(newCoefficients: Array[Double]): this.type = set(weights, newCoefficients)

  val gradMomentum = new DoubleParam(this, "gradMomentum", "Gradient momentum coefficient")
  def getGradMomentum: Double = $(gradMomentum)
  def setGradMomentum(value: Double): this.type = set(gradMomentum, value)

  setDefault(fitIntercept -> true, maxIter -> 1000, lr -> 1e-1, gradMomentum -> 0.9, minGradDelta -> 1e-5)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())
    SchemaUtils.checkColumnTypes(schema, getFeaturesCol, Seq(new VectorUDT(), DoubleType))

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, getOutputCol, new VectorUDT())
    }
  }
}

class LinearRegression(override val uid: String)
  extends Estimator[LinearRegressionModel]
    with LinearRegressionParams
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  private def trainLoop(wBreeze: breeze.linalg.Vector[Double], data: Dataset[(Vector, Double)]): Vector = {
    implicit val vectorEncoder: Encoder[Vector] = ExpressionEncoder()

    val lr = getLr
    val nExamples = data.count().toDouble
    val gradMinDelta = getMinGradDelta
    val momentumVector = Vectors.zeros(wBreeze.length).asBreeze
    for (it <- 1 to getMaxIter) {
      // TODO: are map jobs big enough to deal with job creation overhead? -> Use mapPartitions instead?
      val yGrad: Dataset[Vector] = if ($(fitIntercept))
        data.map(p => {
          val gradTotal = Vectors.zeros(wBreeze.length).asBreeze
          val dLoss = p._1.asBreeze.dot(wBreeze(1 until wBreeze.length)) + wBreeze(0) - p._2
          gradTotal(1 until wBreeze.length) := p._1.asBreeze * dLoss
          gradTotal(0) = dLoss
          Vectors.fromBreeze(gradTotal)
        }).as[Vector]
      else
        data.map(p => Vectors.fromBreeze(p._1.asBreeze * (p._1.asBreeze.dot(wBreeze) - p._2))).as[Vector]

      val grad: Vector =
        Vectors.fromBreeze(yGrad.reduce((a, b) => Vectors.fromBreeze(a.asBreeze +:+ b.asBreeze))
          .asBreeze / nExamples)


      momentumVector *= getGradMomentum
      momentumVector += grad.asBreeze * (1 - getGradMomentum)

      wBreeze -= momentumVector * lr
      if (max(abs(grad.asBreeze)) < gradMinDelta) { // Early stopping
        return Vectors.fromBreeze(wBreeze)
      }
    }
    Vectors.fromBreeze(wBreeze)
  }

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val pairsEncoder: Encoder[(Vector, Double)] = ExpressionEncoder()
    implicit val doubleEncoder: Encoder[Double] = ExpressionEncoder()
    implicit val vectorEncoder: Encoder[Vector] = ExpressionEncoder()

    val data: Dataset[(Vector, Double)] = dataset.select(dataset(getFeaturesCol), dataset(getLabelCol)).as[(Vector, Double)]

    val dim: Int = AttributeGroup
      .fromStructField(dataset.schema($(featuresCol)))
      .numAttributes
      .getOrElse(data.first()._1.size)

    val initWeights = Seq.fill(dim + (if ($(fitIntercept)) 1 else 0))(Random.nextFloat - 0.5)
    setWeights(initWeights.toArray)

    val wBreeze: breeze.linalg.Vector[Double] = Vectors.dense($(weights)).asBreeze

    val newWeights = trainLoop(wBreeze, data)

    setWeights(newWeights.toArray)
    copyValues(new LinearRegressionModel().setParent(this))
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](override val uid: String)
  extends Model[LinearRegressionModel]
    with LinearRegressionParams
    with MLWritable {

  private[made] def this() = this(Identifiable.randomUID("linearRegressionModel"))

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()

    val wBreeze: breeze.linalg.Vector[Double] = Vectors.dense($(weights)).asBreeze
    val dim = wBreeze.length

    val transformUdf = if ($(fitIntercept)) {
      dataset.sqlContext.udf.register(uid + "_predict",
        (x: Vector) => x.asBreeze.dot(wBreeze(1 until dim)) + wBreeze(0)
      )
    } else {
      dataset.sqlContext.udf.register(uid + "_predict",
        (x: Vector) => x.asBreeze.dot(wBreeze)
      )
    }

    dataset.withColumn($(outputCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = super.saveImpl(path)
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)
      val model = new LinearRegressionModel()
      metadata.getAndSetParams(model)
      model
    }
  }
}
