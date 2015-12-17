package eggman89

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by sneha on 12/17/2015.
  */
object CrossValidationTest {

  def main(Args: Array[String]) {
    /*spark stuff*/
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    Logger.getLogger("INFO").setLevel(Level.OFF)
    System.setProperty("hadoop.home.dir", "c:/winutil/")
    val conf = new SparkConf().setAppName("MusicReco").set("spark.serializer", "org.apache.spark.serializer.KryoSerializer").set("spark.executor.memory", "4g").setMaster("local[*]")
    val sc = new SparkContext(conf)

    /*setting up sql context to query the data later on*/
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    println("Spark Context started")
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    Logger.getLogger("INFO").setLevel(Level.OFF)

    val workclass_hm = new hashmap()
    val education_hm = new hashmap()
    val marital_status_hm = new hashmap()
    val occupation_hm = new hashmap()
    val relationship_hm = new hashmap()
    val race_hm = new hashmap()
    val sex_hm = new hashmap()
    val native_country_hm = new hashmap()


    //load data

    val train_set = sc.textFile("dataset/adult.data").map(_.split(",")).map(p => (p(14),

      Vectors.dense((p(0).toDouble), workclass_hm.add(p(1)),
        p(2).toDouble,
        education_hm.add((p(3))),

        p(4).toDouble,
        marital_status_hm.add(p(5)),
        occupation_hm.add(p(6)), relationship_hm.add(p(7)), race_hm.add(p(8)), sex_hm.add(p(9)),
        p(10).toDouble, p(11).toDouble, p(12).toDouble,
        native_country_hm.add(p(13))
      ))).toDF("label", "features")

    val test_set = sc.textFile("dataset/adult.test").map(_.split(",")).map(p => (p(14),
      Vectors.dense((p(0).toDouble), workclass_hm.add(p(1)),
        p(2).toDouble,
        education_hm.add((p(3))),
        p(4).toDouble,
        marital_status_hm.add(p(5)),
        occupation_hm.add(p(6)), relationship_hm.add(p(7)), race_hm.add(p(8)), sex_hm.add(p(9)),
        p(10).toDouble, p(11).toDouble, p(12).toDouble,
        native_country_hm.add(p(13))
      ).toDense))
      .toDF("label", "features")

    /*val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedlabel")
      .fit(train_set)

    //labelIndexer.labels.foreach(println)

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedfeatures")
      .setMaxCategories(50) // features with > 4 distinct values are treated as continuous
      .fit(train_set)

    val gbt = new RandomForestClassifier().setMaxBins(46).setMaxDepth(20).setMaxMemoryInMB(4096)
      .setLabelCol("indexedlabel")
      .setFeaturesCol("indexedfeatures").setNumTrees(5)

  //  println("GBTClassifier parameters:\n" + gbt.explainParams() + "\n")



    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, gbt))

    val paramGrid = new ParamGridBuilder()
    paramGrid.addGrid(gbt.predictionCol, Array( "prediction"))
    paramGrid.addGrid(gbt.labelCol, Array("indexedlabel"))
    paramGrid.addGrid(gbt.featuresCol, Array("features"))
   paramGrid.build().foreach(println)
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid.build())
      .setNumFolds(4) // Use 3+ in practice

    val cvModel = cv.fit(train_set)
    cvModel.transform(test_set.toDF)
    cvModel.transform(test_set)
    val testpreds=cvModel.transform(test_set)
    cvModel.transform(test_set)
    val preds=cvModel.transform(train_set)
    preds.printSchema()
    val ev= new BinaryClassificationEvaluator()
    ev.setLabelCol("indexedlabel")
    ev.evaluate(preds)
    ev.evaluate(testpreds)

    println("Test Output is:" +ev.evaluate(testpreds))
    println("Training Output is:" +ev.evaluate(preds))
*/

  }


}