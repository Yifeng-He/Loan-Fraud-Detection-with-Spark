/*
 * This program is used to determine the result (approved or declined) for a loan application based on
 * the information of the applicant.
 * 
 */
 

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.{Pipeline, PipelineModel}

object LoanFraudDetection {
  
  // function to set the log level
  def setupLogging() = {
    import org.apache.log4j.{Level, Logger}   
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)   
  }
  
  /* the format of each row in the DataFrame. An example of a line is:
  1 	1 	18 	4 	2 	1049 	1 	2 	4 	2 	1 	4 	2 	21 	3 	1 	1 	3 	1 	1 	1
  The columns are:
  0) label (Int), 1) account_balance (categorical), 2) credit_duration (Int), 3) previous_credit (categorical), 
  4) purpose (categorical), 5) amount (float), 6) savings (categorical), 7) employment (categorical), 
  8) installment (categorical), 9) sexMarried (categorical), 10) guarantors (categorical), 
  11) residence_duration(categorical), 12) asset (categorical), 13) age (Int), 14) concurrent_credit (categorical), 
  15) apartment (categorical), 16) number_credits (Int), 17) occupation (categorical) 18) dependents (Int), 
  19) has_phone (categorical), 20) foreign (categorical)
  */
  case class Record(label : Double, raw_features : Vector) // set class label to Double
  
  // the function to convert each line of string into a structured row
  def convertLinetoRow(line: String): Record = {
    val fields = line.split(" ")
    if (fields.size != 21) {
      Record(5, Vectors.zeros(20)) // row to be discarded
    }
    else if (fields(0).trim.toInt > 1) {
      Record(5, Vectors.zeros(20)) // row to be discarded
    }
    else {
      Record(fields(0).trim.toDouble, Vectors.dense(fields(1).trim.toInt, fields(2).trim.toInt, fields(3).trim.toInt, fields(4).trim.toInt, 
        fields(5).trim.toFloat, fields(6).trim.toInt, fields(7).trim.toInt, fields(8).trim.toInt, fields(9).trim.toInt, fields(10).trim.toInt, 
        fields(11).trim.toInt, fields(12).trim.toInt, fields(13).trim.toInt, fields(14).trim.toInt, fields(15).trim.toInt, fields(16).trim.toInt, 
        fields(17).trim.toInt, fields(18).trim.toInt, fields(19).trim.toInt, fields(20).trim.toInt))  
    }
  }
 
   // the main function
   def main(args: Array[String]): Unit = {
    
     // step 1: set up spark session 
     val spark =SparkSession.builder.config(key="spark.sql.warehouse.dir", value="file:///C:/Temp").master("local[*]")
     .appName("LoanFraudDetection").getOrCreate() 
     // set up log level
    setupLogging()
    
    // step 2: load the data into dataFrame
    val dataset_lines = spark.read.textFile("credit_data.txt")
    // convert the DataSet of lines to DataFrame of Rows
    import spark.implicits._
    val data_raw = dataset_lines.map(convertLinetoRow).filter(x => x.label < 2).toDF
    data_raw.printSchema()
    data_raw.groupBy("label").count().show()
    data_raw.show(5)
    
    // step 3: data pre-processing
    // index the categorical columns in the feature vector
    val feature_indexer = new VectorIndexer().setInputCol("raw_features")
      .setOutputCol("indexed_features").setMaxCategories(10)
    val data_indexed = feature_indexer.fit(data_raw).transform(data_raw)
    data_indexed.show(5)  
    // standardize the features to standard normal distribution (mean=0, variance=1)
    val scaler = new StandardScaler().setInputCol("indexed_features").setOutputCol("features")
      .setWithStd(true).setWithMean(true)
    val data_scaled_features = scaler.fit(data_indexed).transform(data_indexed)
    data_scaled_features.show(5)
    // data after pre-processing
    val data = data_scaled_features.select("label", "features")
    
    // step 4: split the data into training and test sets
    val Array(trainingData, testData) = data.randomSplit(Array(0.9, 0.1), seed = 1234)    

    // step 5: construct the pipeline consisting PCA and Random Forest Classifier
    // dimentionality reduction via PCA
    val pca = new PCA().setInputCol("features") 
    // classifier model
    val random_forest = new RandomForestClassifier().setLabelCol("label").setFeaturesCol(pca.getOutputCol)
    // check the parameters descriptions 
    //println(random_forest.explainParams())
    
    // the pipeline
    val pipeline = new Pipeline().setStages(Array(pca, random_forest))
    
    // step 6: grid search over K-fold cross validation for parameter tuning
    // parameter set
    val paramGrid = new ParamGridBuilder()
        .addGrid(pca.k, Array(5, 10, 15))
        .addGrid(random_forest.maxDepth, Array(5, 10)).build()
    // CrossValidator requires: 1) an estimator, 2) a set of ParamMaps, and 3) an evaluator.
    val cross_validator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(4)  // Use 3+ in practice
    // training
    val cvModel = cross_validator.fit(trainingData)
    
    // step 7: prediction and evaluation
    val predictions = cvModel.transform(testData)
    predictions.printSchema()
    predictions.show(1)
    // evaluate
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy = " + accuracy) 

    // step 8: check the learned model
    val bestPipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]  
    val stages = bestPipelineModel.stages  
    val pca_model = stages(0).asInstanceOf[PCAModel]  
    val classifier_model = stages(1).asInstanceOf[RandomForestClassificationModel]   
    println("Best Model Parameters:")  
    println("PCA dimention = " + pca_model.getK)  
    println("Learned classification random forest model:\n" + classifier_model.toDebugString)
         
    spark.stop()
   }  
}


