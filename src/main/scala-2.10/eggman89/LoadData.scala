package eggman89

import eggman89.genreReco.{doNaiveBayes, doDecisionTrees, doLogisticRegressionWithLBFGS, doRandomForest}
import eggman89.hashmap
import org.apache.spark.mllib.stat.Statistics._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.mllib.stat.test.KolmogorovSmirnovTestResult
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.joda.time
import org.joda.time.DateTime

/**
  * Created by snehasis on 12/15/2015.
  */

class hashmapb  extends java.io.Serializable
{
  var obj:Map[String,Int] = Map()
  var id = -1
  def add(value:String): Int ={

    if (obj.contains(value) == true)
    {

      obj(value)
    }

    else
    {
      id = id + 1
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

  def main(args: Array[String])  {

    println("Select a Method to predict")
    println("1: Random Forest; 2:Logistic Regression With LBFGS; 3:Decision Trees;  4:Naive Bayes 5:chiSqTest(other)")
    val method = readInt()

    /*spark stuff*/
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    Logger.getLogger("INFO").setLevel(Level.OFF)
    System.setProperty("hadoop.home.dir", "c:/winutil/")
    val conf = new SparkConf().setAppName("MusicReco").set("spark.serializer", "org.apache.spark.serializer.KryoSerializer").set("spark.executor.memory","4g").setMaster("local[*]")
    val sc = new SparkContext(conf)

    /*setting up sql context to query the data later on*/
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    println("Spark Context started")
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    Logger.getLogger("INFO").setLevel(Level.OFF)




    /*create hash-tables for non numeric attributes*/

    val workclass_hm = new hashmap()
    val education_hm = new hashmap()
    val marital_status_hm = new hashmap()
    val occupation_hm = new hashmap()
    val relationship_hm = new hashmap()
    val race_hm = new hashmap()
    val sex_hm = new hashmapb()
    val native_country_hm = new hashmap()
    val sal_50k_hm = new hashmapb()

    //load data

    val train_set = sc.textFile("dataset/adult_data.txt").map(_.split(",")).map(p=> LabeledPoint(p(14).drop(1).toString.toInt,

      Vectors.dense((p(0).toDouble),workclass_hm.add(p(1)) ,
        p(2).toDouble,
        education_hm.add((p(3))),

        p(4).toDouble,
        marital_status_hm.add(p(5)),
        occupation_hm.add(p(6)), relationship_hm.add(p(7)), race_hm.add(p(8)),  sex_hm.add(p(9)),
        p(10).toDouble, p(11).toDouble, p(12).toDouble,
        native_country_hm.add(p(13))
      )))

    val test_is_train = sc.textFile("dataset/adult_test.txt").map(_.split(",")).map(p=>( 2 ,
      Vectors.dense((p(0).toDouble),workclass_hm.add(p(1)) ,
       p(2).toDouble,
        education_hm.add((p(3))),
        p(4).toDouble,
        marital_status_hm.add(p(5)),
        occupation_hm.add(p(6)), relationship_hm.add(p(7)), race_hm.add(p(8)),  sex_hm.add(p(9)),
        p(10).toDouble, p(11).toDouble, p(12).toDouble,
        native_country_hm.add(p(13))
      ).toDense,sal_50k_hm.add(p(14))
      )
    )

    var it : Int = 0;


    var predicted_res_RDD  : RDD[(Int, Int, Int)] = sc.emptyRDD

    if (method == 1)
    {
      predicted_res_RDD = doRandomForest.test(doRandomForest.train(train_set,10,32,40),test_is_train)


    }

    if(method ==2)
    {
      predicted_res_RDD = doLogisticRegressionWithLBFGS.test(doLogisticRegressionWithLBFGS.train(train_set),test_is_train)
    }

    if(method ==3)
    {
      predicted_res_RDD = doDecisionTrees.test(doDecisionTrees.train(train_set,29,32),test_is_train)
    }

    if(method ==4)
    {
      predicted_res_RDD = doNaiveBayes.test(doNaiveBayes.train(train_set,1),test_is_train)
    }
    if(method ==5)
    {
     // chiSqTest.do_test(RDD_LP_trainset)

    }

   // predicted_res_RDD.foreach(println)
    val predictionAndLabels : RDD[(Double,Double)] = predicted_res_RDD.toDF().map(l => (l(1).toString.toDouble,l(2).toString.toDouble))
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    println(metrics.confusionMatrix.toString())
    println("Precision = " + precision)
    println("End: Prediction")

  //  Statistics_.


  }

}
