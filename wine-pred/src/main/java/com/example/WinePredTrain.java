package com.example;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.classification.RandomForestClassifier;

import java.io.IOException;
import java.util.Arrays;

public class WinePredTrain {

    public static Dataset<Row> loadAndProcessData(SparkSession spark, String filePath) {
        // rreading the file
        Dataset<Row> df = spark.read()
                .option("header", "true")
                .option("sep", ";")
                .option("inferSchema", "true")
                .csv(filePath);

                // renaming coumn names
        String[] columns = new String[] { "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol",
                "quality" };
        df = df.toDF(columns);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(Arrays.copyOfRange(columns, 0, columns.length - 1))
                .setOutputCol("features");
        StandardScaler scaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setWithStd(true).setWithMean(true);

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] { assembler, scaler });
        
        PipelineModel pipelineModel = pipeline.fit(df);
        
        // returning preprocessed dataset
        return pipelineModel.transform(df);
    }
            
    public static void main(String[] args) throws IOException {
        SparkSession spark = SparkSession.builder()
                .appName("Wine-Pred-Training")
                .getOrCreate();

        // Loadoing and processing datasets
        Dataset<Row> trainData = loadAndProcessData(spark, "s3://wine-pred/TrainingDataset.csv");
        Dataset<Row> validationData = loadAndProcessData(spark, "s3://wine-pred/ValidationDataset.csv");

        // Applying Random Forest Model
        RandomForestClassifier rf = new RandomForestClassifier().setLabelCol("quality").setFeaturesCol("scaledFeatures");

        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();

        // finetuning
        ParamMap[] paramGrid = paramGridBuilder
                .addGrid(rf.numTrees(), new int[] { 10, 20 })
                .addGrid(rf.maxDepth(), new int[] { 4, 8 })
                .build();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("quality")
        .setPredictionCol("prediction");

        CrossValidator cv = new CrossValidator().setEstimator(rf).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid)
                .setNumFolds(3);

        // training with cross validation
        CrossValidatorModel cvModel = cv.fit(trainData);
        double bestScore = cvModel.avgMetrics()[0];

        System.out.println("F1 Score: " + bestScore);

        String bestHyperparametersModelPath = "s3://wine-pred/top_model";
        RandomForestClassificationModel bestGbtModel = (RandomForestClassificationModel) cvModel.bestModel();
        bestGbtModel.write().overwrite().save(bestHyperparametersModelPath);
        System.out.println("Top model with finetuned hyperparermeters is saved to " + bestHyperparametersModelPath);

        spark.stop();
    }

}
