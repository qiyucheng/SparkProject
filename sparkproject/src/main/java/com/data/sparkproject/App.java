package com.data.sparkproject;
import scala.Tuple2;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.DoubleFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.GeneralizedLinearModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LassoModel;
import org.apache.spark.mllib.regression.LassoWithSGD;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.mllib.regression.RidgeRegressionModel;
import org.apache.spark.mllib.regression.RidgeRegressionWithSGD;
import org.apache.spark.SparkConf;

/**
 * Build a prediction model by RegressionModel
 *
 */
public class App 
{
	public static void main(String[] args) {
	    SparkConf sparkConf = new SparkConf()
	          .setAppName("Regression")
	          .setMaster("local[2]");
	    JavaSparkContext sc = new JavaSparkContext(sparkConf);
	    JavaRDD<String> data = sc.textFile("C://Users//Administrator//workspace//lpsa.data");
	    JavaRDD<LabeledPoint> parsedData = data.map(
	    	      new Function<String, LabeledPoint>() {
	    	        public LabeledPoint call(String line) {
	    	          String[] parts = line.split(",");
	    	          String[] features = parts[1].split(" ");
	    	          double[] v = new double[features.length];
	    	          for (int i = 0; i < features.length - 1; i++)
	    	            v[i] = Double.parseDouble(features[i]);
	    	          return new LabeledPoint(Double.parseDouble(parts[0]), Vectors.dense(v));
	    	        }
	    	      }
	    	    );
	    	    parsedData.cache();
	 
	    int numIterations = 100; //迭代次数
	    LinearRegressionModel model = LinearRegressionWithSGD.train(parsedData.rdd(), numIterations);
	    RidgeRegressionModel model1 = RidgeRegressionWithSGD.train(parsedData.rdd(), numIterations);
	    LassoModel model2 = LassoWithSGD.train(parsedData.rdd(), numIterations);
	 
	    print(parsedData, model);
	    print(parsedData, model1);
	    print(parsedData, model2);
	 
	    //预测一条新数据方法
	    double[] d = new double[]{1.0, 1.0, 2.0, 1.0, 3.0, -1.0, 1.0, -2.0};
	    Vector v = Vectors.dense(d);
	    System.out.println(model.predict(v));
	    System.out.println(model1.predict(v));
	    System.out.println(model2.predict(v));
	}
	 
	public static void print(JavaRDD<LabeledPoint> parsedData, final GeneralizedLinearModel model) {
	    JavaPairRDD<Double, Double> valuesAndPreds = parsedData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
			public Tuple2<Double, Double> call(LabeledPoint point) throws Exception {
			    double prediction = model.predict(point.features()); //用模型预测训练数据
			    return new Tuple2<>(point.label(), prediction);
			}
		});
	 
	    Double MSE = valuesAndPreds.mapToDouble(new DoubleFunction<Tuple2<Double, Double>>() {
			@Override
			public double call(Tuple2<Double, Double> t) throws Exception {
				return Math.pow(t._1() - t._2(), 2);
			}
		}).mean(); //计算预测值与实际值差值的平方值的均值
	    System.out.println(model.getClass().getName() + " training Mean Squared Error = " + MSE);
	    System.out.println("----------------------算法结束------------------------");
	}

}
