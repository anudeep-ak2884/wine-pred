package com.example;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.classification.RandomForestClassificationModel;

import java.util.Arrays;
import com.example.WinePredTrain;

public class WinePredTest {

    public static void main(String[] args) {
        // Processing cli arguments
        String modelPath = args.length > 0 ? args[0] : "s3://wine-pred/top_model/";
        String testFilePath = args.length > 1 ? args[1] : "s3://wine-pred/ValidationDataset.csv";

        // Initializing Spark Session
        SparkSession spark = SparkSession
                .builder()
                .appName("Wine-Pred-Testing")
                .getOrCreate();

        // Loading dataset
        Dataset<Row> validationData = WinePredTrain.loadAndProcessData(spark, testFilePath);
        validationData.select("scaledFeatures").show(false);

        // Loading the top saved model
        RandomForestClassificationModel loadedModel = RandomForestClassificationModel.load(modelPath);
        Dataset<Row> predictions = loadedModel.transform(validationData);

        // printing
        predictions.show();

        // Evaluating the model
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double score = evaluator.evaluate(predictions);
        System.out.println("f1 score : " + score);

        spark.stop();
    }

}
