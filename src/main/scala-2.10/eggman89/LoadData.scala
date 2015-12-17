package eggman89


import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.feature._

class hashmap  extends java.io.Serializable
{
  var obj:Map[String,Int] = Map()
  var id = -1
  def add(value:String): Int ={

    if (obj.contains(value) == true)
    {
      if (value == "?")
      {
        return 0;
      }
      obj(value)
    }

    else
    {      id = id + 1

      obj = obj +(value->id)
      id
    }
  }

  def findval(value : Int) : String = {
    val default = ("-1",0)
    obj.find(_._2==value).getOrElse(default)._1
  }
}


object LoadData {

  def main(args: Array[String]) {


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

    val train_set = sc.textFile("dataset/adult.data").map(_.split(",")).map(p=> (p(14),

      Vectors.dense((p(0).toDouble),workclass_hm.add(p(1)) ,
        p(2).toDouble,
        education_hm.add((p(3))),

        p(4).toDouble,
        marital_status_hm.add(p(5)),
        occupation_hm.add(p(6)), relationship_hm.add(p(7)), race_hm.add(p(8)),  sex_hm.add(p(9)),
        p(10).toDouble, p(11).toDouble, p(12).toDouble,
        native_country_hm.add(p(13))
      ))).toDF("salary", "attributes")

    val test_set = sc.textFile("dataset/adult.test").map(_.split(",")).map(p=>( p(14),
      Vectors.dense((p(0).toDouble),workclass_hm.add(p(1)) ,
        p(2).toDouble,
        education_hm.add((p(3))),
        p(4).toDouble,
        marital_status_hm.add(p(5)),
        occupation_hm.add(p(6)), relationship_hm.add(p(7)), race_hm.add(p(8)),  sex_hm.add(p(9)),
        p(10).toDouble, p(11).toDouble, p(12).toDouble,
        native_country_hm.add(p(13))
      ).toDense))
      .toDF("salary", "attributes")

    val labelIndexer = new StringIndexer()
      .setInputCol("salary")
      .setOutputCol("indexedSalary")
      .fit(train_set)

    //labelIndexer.labels.foreach(println)

    val featureIndexer = new VectorIndexer()
      .setInputCol("attributes")
      .setOutputCol("indexedAttributes")
      .setMaxCategories(50) // features with > 4 distinct values are treated as continuous
      .fit(train_set)

    val dt = new GBTClassifier().setMaxBins(46).setMaxDepth(20).setMaxMemoryInMB(4096)
      .setLabelCol("indexedSalary")
      .setFeaturesCol("indexedAttributes").setMaxIter(15)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedSal")
      .setLabels(labelIndexer.labels)

    // Chain indexers and tree in a Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    // Train model.  This also runs the indexers.
    val model = pipeline.fit(train_set)

    // Make predictions.
    val predictions = model.transform(test_set)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedSalary")
      .setPredictionCol("prediction")
      .setMetricName("precision")

    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    //val treeModel = model.stages(2).asInstanceOf[RandomForestModel]
  //  println("Learned classification tree model:\n" + treeModel.toDebugString)


  }
}
