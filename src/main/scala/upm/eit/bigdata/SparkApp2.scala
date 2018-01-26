package upm.eit.bigdata

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.evaluation.RegressionMetrics
import scala.collection.mutable.ArrayBuffer

object SparkApp2 {

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder()
      .appName("Assignment1")
      .master("local[4]") //can put with vm option -Dspark.master=local in run configuration
      .config("spark.executor.memory", "6g")
      .getOrCreate()
    import spark.implicits._
    spark.sparkContext.setLogLevel("WARN")
    var df = spark.read
      .option("delimiter", ",")
      .option("header", true) //headers not visible as the first row but as headers
      .csv("./src/main/resources/input.csv")
      .drop("Year", "ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "DayOfMonth", "DepTime", "CRSDepTime", "TailNum", "Origin", "TaxiOut") //columns removed
      .withColumn("Month", col("Month").cast("int"))
      .withColumn("DayOfWeek", col("DayOfWeek").cast("int"))
      .withColumn("CRSArrTime", col("CRSArrTime").cast("int"))
      .withColumn("DepDelay", col("DepDelay").cast("int"))
      .withColumn("FlightNum", col("FlightNum").cast("int"))
      .withColumn("CRSElapsedTime", col("CRSElapsedTime").cast("int"))
      .withColumn("ArrDelay", col("ArrDelay").cast("int"))
      .withColumn("Distance", col("Distance").cast("int"))

    df = df.filter(col("Cancelled") === 0); //only flights that were realized
    df = df.drop("Cancelled", "CancellationCode"); //no longer needed
    df = df.filter(!col("ArrDelay").isNull); // after string->int conversion NA values are nulls and we remove them
//    df = df.limit(100000) // for testing of a smaller subset
    if (df.count() > 5000000) // due to the limited performance of the computer we limit the number of records to 5 million, feel free to comment this section
      {
        df = df.sample(true, 1D*5000000/df.count)
      }

    var expre = expr("(((CRSArrTime - CRSArrTime % 100) / 100) * 60 + CRSArrTime % 100 + DepDelay) BETWEEN 360 AND 1321 ") //var indicating predicted arrival time including the delay at start, where 1=day (from 6 to 22) and 0=night (from 22 to 6)
    df = df.withColumn("DayArrival", expre.cast("int"))

    /////////////////////
    //GLM
    //FACTORS
    //month
    var encoder = new OneHotEncoder()
      .setInputCol("Month")
      .setOutputCol("MonthFactor")
    df = encoder.transform(df)

    //dayofweek
    encoder = new OneHotEncoder()
      .setInputCol("DayOfWeek")
      .setOutputCol("DayOfWeekFactor")
    df = encoder.transform(df)

    //uniquecarrier
    var indexer = new StringIndexer()
      .setInputCol("UniqueCarrier")
      .setOutputCol("UniqueCarrierIndexed")

    var indexerModel = indexer.fit(df)
    // Create new column "indexed" with categorical values transformed to indices
    var indexedData = indexerModel.transform(df)
    encoder = new OneHotEncoder()
      .setInputCol("UniqueCarrierIndexed")
      .setOutputCol("UniqueCarrierFactor")
    df = encoder.transform(indexedData)
//
//    //flightnum - removed due to almost 10000factors
//    encoder = new OneHotEncoder()
//      .setInputCol("FlightNum")
//      .setOutputCol("FlightNumFactor")
//    df = encoder.transform(df)

    //Dest
    indexer = new StringIndexer()
      .setInputCol("Dest")
      .setOutputCol("DestIndexed")
    indexerModel = indexer.fit(df)
    // Create new column "indexed" with categorical values transformed to indices
    indexedData = indexerModel.transform(df)
    encoder = new OneHotEncoder()
      .setInputCol("DestIndexed")
      .setOutputCol("DestFactor")
    df = encoder.transform(indexedData)

    //////////////////////////
    val assembler = new VectorAssembler()
      .setInputCols(Array("MonthFactor", "DayOfWeekFactor", "DayArrival", "UniqueCarrierFactor", "CRSElapsedTime", "DepDelay", "Distance", "DestFactor"))//, "FlightNumFactor")) // removed due to almost 10000 different results
      .setOutputCol("features")

    val output = assembler.transform(df)
    val glr = new GeneralizedLinearRegression()
      .setFamily("gaussian")
      .setLink("identity")
      .setMaxIter(10)
      .setRegParam(0.3)
      .setFeaturesCol("features")   // setting features column
      .setLabelCol("ArrDelay")       // setting label column

    // Fit the model
    val model = glr.fit(output)

    //80-20 estimation
    //numOfIter times
    var r2Array = ArrayBuffer[Double]()
    val numOfIter = 1 // change to the number of tries you prefer i.e. numOfIter = 5 // for this computer after 2 it caused problems, seedArray should be changed accordingly
    val r = scala.util.Random // random seed changing for numOfIter > 1
    for( i <- 0 to numOfIter-1) {

      val splits = output.randomSplit(Array(0.8, 0.2), seed = r.nextLong())
      val train = splits(0).cache()
      val test = splits(1).cache()
      val model = glr.fit(train)

      val result = model.transform(test).select("ArrDelay", "prediction").withColumn("ArrDelay", col("ArrDelay").cast("Double")) //2columns needed for comparison
      var myRDD = result.map(r => (r(0).asInstanceOf[Double], r(1).asInstanceOf[Double])).rdd
//      myRDD.take(20).foreach(println) //print first 20 predictions and real delays
      val regressionMetrics = new RegressionMetrics(myRDD)
      println(s"r^2: ${regressionMetrics.r2}")
      println(s"MSE: ${regressionMetrics.meanSquaredError}")
      println(s"RMSE: ${regressionMetrics.rootMeanSquaredError}")
      r2Array += regressionMetrics.r2
    }

    val r2Final = r2Array.sum/r2Array.length //mean of tries
    println(r2Final)

    // Print the coefficients and intercept for generalized linear regression model
    println(s"Coefficients: ${model.coefficients}")
    println(s"Intercept: ${model.intercept}")

    // Summarize the model over the training set and print out some metrics
    val summary = model.summary
    println(s"Coefficient Standard Errors: ${summary.coefficientStandardErrors.mkString(",")}")
    println(s"T Values: ${summary.tValues.mkString(",")}")
    println(s"P Values: ${summary.pValues.mkString(",")}")
    println(s"Dispersion: ${summary.dispersion}")
    println(s"Null Deviance: ${summary.nullDeviance}")
    println(s"Residual Degree Of Freedom Null: ${summary.residualDegreeOfFreedomNull}")
    println(s"Deviance: ${summary.deviance}")
    println(s"Residual Degree Of Freedom: ${summary.residualDegreeOfFreedom}")
    println(s"AIC: ${summary.aic}")
    println("Deviance Residuals: ")
    summary.residuals().show()

  }
}