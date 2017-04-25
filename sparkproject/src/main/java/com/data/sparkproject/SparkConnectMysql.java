package com.data.sparkproject;
import java.util.Properties;  
import org.apache.spark.SparkConf;  
import org.apache.spark.api.java.JavaSparkContext;  
import org.apache.spark.sql.DataFrame;  
import org.apache.spark.sql.SQLContext;  
import org.apache.spark.sql.SaveMode;
import org.apache.log4j.Logger;
  
/**
 * Top level model for connect mysql from spark api.
 */ 
public class SparkConnectMysql {
	private static Logger logger = Logger.getLogger(SparkConnectMysql.class);
	public static void main(String[] args) {
		SparkConf sparkConf = new SparkConf();
		sparkConf.setAppName("SparkConnectMysql");
		sparkConf.setMaster("local[2]");
		JavaSparkContext sc = null;  
		try {
			sc = new JavaSparkContext(sparkConf);
			SQLContext sqlContext = new SQLContext(sc);
			String url = "jdbc:mysql://10.1.66.161:3306/dataming";  
			String table_input = "feature_min";  
			Properties connectionProperties = new Properties();  
			connectionProperties.setProperty("dbtable", table_input);// 设置表  
			connectionProperties.setProperty("user", "root");// 设置用户名  
			connectionProperties.setProperty("password", "123456");// 设置密码   
			// 读取数据  
			DataFrame jdbcDF = sqlContext.read().jdbc(url, table_input,connectionProperties);
			jdbcDF.printSchema();
			// 写入数据  
			String url2 = "jdbc:mysql://10.1.66.161:3306/dataming";  
			Properties connectionProperties2 = new Properties();  
			connectionProperties2.setProperty("user", "root");// 设置用户名  
			connectionProperties2.setProperty("password", "123456");// 设置密码  
			String table_output = "predict_order_result";
			// SaveMode.Append表示添加的模式
			// SaveMode.Append:在数据源后添加
			// SaveMode.Overwrite:如果如果数据源已经存在记录，则覆盖
			// SaveMode.ErrorIfExists:如果如果数据源已经存在记录，则包异常
			// SaveMode.Ignore:如果如果数据源已经存在记录，则忽略 
			jdbcDF.write().mode(SaveMode.Append).jdbc(url2, table_output, connectionProperties2);
			
			} catch (Exception e) {  
				logger.error("|main|exception error", e);  
			} finally {  
				if (sc != null) {  
					sc.stop();  
				}
			}
		}  
}  
