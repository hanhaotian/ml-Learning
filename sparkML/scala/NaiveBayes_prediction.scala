  def try2Prediction() : Unit= {
    val sc = new SparkContext(new SparkConf().setAppName("try2Prediction"))
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    //val lm = new SparkContext(new SparkConf().setAppName("try2Load"))

    val model = NaiveBayesModel.load(sc,"/work/hthan4/music/data/spark/modle")
    import sqlContext.implicits._

    val dataDF =sc.textFile("").map(data=>{
      val json = JSONObjectEx.fromObject(data)
      if (null == json) DataRecord("","")
      else {
        val title = json.getStringByKeys("name")
        DataRecord(title,HanLPSpliter.getInstance().seg(title).toArray().map(_.toString).toList.mkString(""," ",""))
      }
    }).filter(f=> 3 < f.data.toString.length).toDF()

    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val wordsData = tokenizer.transform(dataDF)

    val hashingTF = new HashingTF().setNumFeatures(500000).setInputCol("words").setOutputCol("rawFeatures")
    val featurizedData = hashingTF.transform(wordsData)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)

    val DataRdd = rescaledData.select($"data",$"features").map {
      case Row(data: String, features: Vector) =>
        (data, Vectors.dense(features.toArray))
    }
   val res = DataRdd.map(m=>(m._1,model.predict(m._2))).filter(_._2 == 0).collect()//sortBy(_._2).collect().reverse//

    sc.parallelize(res.map(f=>s"${f._1}\t${f._2}")).saveAsTextFile(HDFSFileUtil.clean(""))

    sc.stop()
  }
  
  case class DataRecord(data: String, text: String)

  case class RawDataRecord(category: String, text: String)
