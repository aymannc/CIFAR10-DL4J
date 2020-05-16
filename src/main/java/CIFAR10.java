import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class CIFAR10 {
    static String basePath = "C:\\datasets\\CIFAR-10";
    static int height = 32;
    static int width = 32;
    static int channels = 3;
    static int outputNum = 10;
    static int batchSize = 256;
    static int epochCount = 30;
    static int seed = 98;
    static int labelIndex = 1;
    static double learningRate = 0.001;

    public static void main(String[] args) throws IOException {

        MultiLayerNetwork model;
        try {
            System.out.println("Loading The model");
            model = ModelSerializer.restoreMultiLayerNetwork(new File(basePath + "\\model.zip"));
        } catch (IOException ignored) {
            CIFAR10 cifar10 = new CIFAR10();
            model = cifar10.buildModel();
        }


        Random randomGenNum = new Random(seed);
        System.out.println("Loading training Data");
        File trainDataFile = new File(basePath + "\\train");
        FileSplit trainFileSplit = new FileSplit(trainDataFile, NativeImageLoader.ALLOWED_FORMATS, randomGenNum);
        ParentPathLabelGenerator labelMarker = new ParentPathLabelGenerator();
        ImageRecordReader trainImageRecordReader = new ImageRecordReader(height, width, channels, labelMarker);
        trainImageRecordReader.initialize(trainFileSplit);
        DataSetIterator trainDataSetIterator = new
                RecordReaderDataSetIterator(trainImageRecordReader, batchSize, labelIndex, outputNum);
        DataNormalization scalar = new ImagePreProcessingScaler(0, 1);
        trainDataSetIterator.setPreProcessor(scalar);

        System.out.println("Loading testing Data");
        File testDataFile = new File(basePath + "\\test");
        FileSplit testFileSplit = new FileSplit(testDataFile, NativeImageLoader.ALLOWED_FORMATS, randomGenNum);
        ImageRecordReader testImageRecordReader = new ImageRecordReader(height, width, channels, labelMarker);
        testImageRecordReader.initialize(testFileSplit);

        DataSetIterator testDataSetIterator = new
                RecordReaderDataSetIterator(testImageRecordReader, batchSize, labelIndex, outputNum);
        testDataSetIterator.setPreProcessor(scalar);

        UIServer uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));


        System.out.println("Total params:" + model.numParams());
        Evaluation evaluation = model.evaluate(testDataSetIterator);
        double startingAccuracy = evaluation.accuracy();
        System.out.println("Starting accuracy " + startingAccuracy);
        for (int i = 0; i < epochCount; i++) {
            testDataSetIterator.reset();
            trainDataSetIterator.reset();


            System.out.println("old accuracy " + startingAccuracy);
            System.out.println("new accuracy " + evaluation.accuracy());

            System.out.println("Epoch " + (i + 1));
            model.fit(trainDataSetIterator);
            evaluation = model.evaluate(testDataSetIterator);
            System.out.println(evaluation.stats());
            if (evaluation.accuracy() > startingAccuracy) {
                startingAccuracy = evaluation.accuracy();
                System.out.println("Saving model !");
                ModelSerializer.writeModel(model, new File(basePath + "\\model.zip"), true);
            }
        }
    }

    public MultiLayerNetwork buildModel() {
        System.out.println("Building the model...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(learningRate))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).activation(Activation.LEAKYRELU)
                        .nIn(channels).nOut(32).build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder().kernelSize(2, 2).stride(2, 2).poolingType(SubsamplingLayer.PoolingType.MAX).build())

                .layer(new ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).padding(1, 1).activation(Activation.LEAKYRELU)
                        .nOut(64).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).activation(Activation.LEAKYRELU)
                        .nOut(64).build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder().kernelSize(2, 2).stride(2, 2).poolingType(SubsamplingLayer.PoolingType.MAX).build())

                .layer(new ConvolutionLayer.Builder().kernelSize(1, 1).stride(1, 1).padding(1, 1).activation(Activation.LEAKYRELU)
                        .nOut(64).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1).activation(Activation.LEAKYRELU)
                        .nOut(128).build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder().kernelSize(2, 2).stride(2, 2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
                .layer(new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(500)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(outputNum)
                        .dropOut(0.5)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }
}
