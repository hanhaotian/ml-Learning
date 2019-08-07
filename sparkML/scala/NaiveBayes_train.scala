  def testSparkMath() : Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("testSparkMath"))
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    //引入sqlContext隐式转换  rdd自动转df
    import sqlContext.implicits._

    val srcRDD = sc.textFile("",30).map {
      x =>
        val data = x.split("\t")
        val res = HanLPSpliter.getInstance().seg(data(0))
        RawDataRecord(data(1),res.toArray().map(_.toString).toList.mkString(""," ",""))
    }

    //70%作为训练数据，30%作为测试数据
    val splits = srcRDD.randomSplit(Array(0.7, 0.3))
    val trainingDF = splits(0).toDF()
    val testDF = splits(1).toDF()

    //将词语转换成数组
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

    val wordsData = tokenizer.transform(trainingDF)

    println("output1：")
    wordsData.select($"category",$"text",$"words").take(1).foreach(println)

    //计算每个词在文档中的词频
    val hashingTF = new HashingTF().setNumFeatures(500000).setInputCol("words").setOutputCol("rawFeatures")
    val featurizedData = hashingTF.transform(wordsData)
    println("output2：")
    val output2 = featurizedData.select($"category", $"words", $"rawFeatures").take(1).toString
    println(output2)

    //计算每个词的TF-IDF
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)

    println("output3：")
    val output3 = rescaledData.select($"category", $"features").take(1).toString
    println(output3)

    //转换成Bayes的输入格式
    val trainDataRdd = rescaledData.select($"category",$"features").map {
      case Row(label: String, features: Vector) =>
        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
    }

    println("output4：")
    val output4 = trainDataRdd.take(1).toString
    println(output4)
    //训练模型
    val model = NaiveBayes.train(trainDataRdd, lambda = 1.0, modelType = "multinomial")
    //测试数据集，做同样的特征表示及格式转换
    val testwordsData = tokenizer.transform(testDF)
    val testfeaturizedData = hashingTF.transform(testwordsData)
    val testrescaledData = idfModel.transform(testfeaturizedData)
    val testDataRdd = testrescaledData.select($"category",$"features").map {
      case Row(label: String, features: Vector) =>
        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
    }

    //对测试数据集使用训练模型进行分类预测
    val testpredictionAndLabel = testDataRdd.map(p => (model.predict(p.features), p.label))

    //统计分类准确率
    val testaccuracy = 1.0 * testpredictionAndLabel.filter(x => x._1 == x._2).count() / testDataRdd.count()
    println("output5：")
    println(testaccuracy)
    sc.stop()
    val sm = new SparkContext(new SparkConf().setAppName("saveSparkMath"))

    model.save(sm,HDFSFileUtil.clean(""))
    sm.stop()
  }
