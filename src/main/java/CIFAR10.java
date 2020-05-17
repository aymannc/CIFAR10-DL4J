import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
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
    private final String mode;
    private MultiLayerNetwork model;


    public static void main(String[] args) throws IOException {

        CIFAR10 cifar10 = new CIFAR10("build");

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();
        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        InMemoryStatsStorage inMemoryStatsStorage = new InMemoryStatsStorage();
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(inMemoryStatsStorage);
        //Then add the StatsListener to collect this information from the network, as it trains
        cifar10.getModel().setListeners(new StatsListener(inMemoryStatsStorage));

        cifar10.trainModelAndEvaluate();


    }

    public CIFAR10(String mode) {
        this.mode = mode;
        try {
            if (this.mode.equals("load")) {
                System.out.println("Loading The model");
                this.model = ModelSerializer.restoreMultiLayerNetwork(new File(basePath + "\\model.zip"));
            } else if (this.mode.equals("build")) {
                this.buildModel();
            }
        } catch (IOException ignored) {
            this.buildModel();
        }

    }

    public MultiLayerNetwork getModel() {
        return model;
    }

    public void buildModel() {
        System.out.println("Building the model...");
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
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

        this.model = new MultiLayerNetwork(configuration);
        this.model.init();
    }

    public DataSetIterator getDataSetIterator(String mode) throws IOException {


        File dataFile;
        if (mode.equals("train")) {
            dataFile = new File(basePath + "\\train");
        } else {
            dataFile = new File(basePath + "\\test");
        }
        Random randomGenNum = new Random(seed);
        System.out.println("Loading " + mode + " Data");
        FileSplit trainFileSplit = new FileSplit(dataFile, NativeImageLoader.ALLOWED_FORMATS, randomGenNum);
        ParentPathLabelGenerator labelMarker = new ParentPathLabelGenerator();
        ImageRecordReader imageRecordReader = new ImageRecordReader(height, width, channels, labelMarker);
        imageRecordReader.initialize(trainFileSplit);
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, batchSize, labelIndex, outputNum);
        DataNormalization scalar = new ImagePreProcessingScaler(0, 1);
        dataSetIterator.setPreProcessor(scalar);

        return dataSetIterator;
    }

    public void trainModelAndEvaluate() throws IOException {

        DataSetIterator trainDataSetIterator = this.getDataSetIterator("train");
        DataSetIterator testSetIterator = this.getDataSetIterator("test");

        Evaluation evaluation;
        double startingAccuracy = 0;
        if (this.mode.equals("load")) {
            evaluation = model.evaluate(testSetIterator);
            startingAccuracy = evaluation.accuracy();
            System.out.println("Starting accuracy " + startingAccuracy);
        }
        int i = 1;
        while (startingAccuracy < 1.0) {
            testSetIterator.reset();
            trainDataSetIterator.reset();


            System.out.println("Epoch " + (i++));
            model.fit(trainDataSetIterator);
            evaluation = model.evaluate(testSetIterator);
            System.out.println("old accuracy " + startingAccuracy);
            System.out.println("new accuracy " + evaluation.accuracy());
            if (evaluation.accuracy() > startingAccuracy) {
                startingAccuracy = evaluation.accuracy();
                System.out.println("Saving model !");
                ModelSerializer.writeModel(model, new File("model.zip"), true);
            }

            System.out.println(evaluation.stats());
        }
    }
}
