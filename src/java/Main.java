import java.io.IOException;
import java.util.ArrayList;

public class Main {

    private static final String TRAIN_IMAGES_PATH = "data/train-images-idx3-ubyte";
    private static final String TRAIN_LABELS_PATH = "data/train-labels-idx1-ubyte";
    private static final String TEST_IMAGES_PATH = "data/t10k-images-idx3-ubyte";
    private static final String TEST_LABELS_PATH = "data/t10k-labels-idx1-ubyte";

    private static final int INPUT_SIZE = 28 * 28;
    private static final int[] HIDDEN_LAYER_SIZES = {256, 128};
    private static final int OUTPUT_SIZE = 10;

    private static final String WEIGHTS_PATH = "weights/pesos.csv";
    private static final String LOSS_LOG_PATH = "weights/mse_values.txt";

    private static final int TRAIN_EXAMPLES = 60000;
    private static final int TEST_EXAMPLES = 10000;

    public static void main(String[] args) {
        boolean shouldTrain = true;
        if (args.length > 0 && "--test-only".equalsIgnoreCase(args[0])) {
            shouldTrain = false;
        }

        Main mainInstance = new Main();
        mainInstance.trainAndTest(shouldTrain);
    }

    public void trainAndTest(boolean shouldTrain) {

        double lossThreshold = 0.005;
        double learningRate = 0.1;

        System.out.println("========================================");
        System.out.println("  TREINO DA REDE NEURAL - MNIST");
        System.out.println("========================================");
        System.out.println("Arquitetura: " + architectureString());
        System.out.println("Inputs: " + INPUT_SIZE + " pixels (28x28)");
        System.out.println("Classes: " + OUTPUT_SIZE + " (dígitos 0-9)");
        System.out.println("Learning Rate: " + learningRate);
        System.out.println("Loss Threshold: " + lossThreshold);
        System.out.println("========================================\n");

        MnistLoader.MnistDataset trainingData = null;
        MnistLoader.MnistDataset testData;

        try {
            if (shouldTrain) {
                trainingData = MnistLoader.load(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, TRAIN_EXAMPLES);
            }
            testData = MnistLoader.load(TEST_IMAGES_PATH, TEST_LABELS_PATH, TEST_EXAMPLES);
        } catch (IOException e) {
            System.err.println("Erro ao carregar MNIST: " + e.getMessage());
            System.err.println("Execute python download_mnist.py para descarregar os ficheiros IDX para a pasta data/.");
            return;
        }

        if (shouldTrain && trainingData != null) {
            System.out.println("Dados de treino: " + trainingData.getImages().length + " imagens");
        } else {
            System.out.println("Modo teste: a carregar apenas os dados de teste.");
        }
        System.out.println("Dados de teste: " + testData.getImages().length + " imagens\n");

        double[][] trainingTargets = shouldTrain && trainingData != null
                ? toOneHot(trainingData.getLabels(), OUTPUT_SIZE)
                : new double[0][0];

        ArrayList<Layer> layers = buildNetwork();
        NeuralNetwork nn = new NeuralNetwork(layers);

        if (shouldTrain && trainingData != null) {
            long startTime = System.currentTimeMillis();
            nn.train(trainingData.getImages(), trainingTargets, lossThreshold, learningRate, LOSS_LOG_PATH, WEIGHTS_PATH);
            long endTime = System.currentTimeMillis();
            long trainingTime = endTime - startTime;

            System.out.println("\n========================================");
            System.out.println("  TREINO CONCLUÍDO");
            System.out.println("========================================");
            System.out.println("Tempo de treino: " + trainingTime + " ms (" + (trainingTime / 1000.0) + " segundos)");
            System.out.println("========================================\n");
        } else {
            System.out.println("A saltar treino. A carregar pesos existentes de weights/pesos.csv...");
            nn.loadWeights(WEIGHTS_PATH);
        }

        System.out.println("========================================");
        System.out.println("  TESTE DA REDE NEURAL");
        System.out.println("========================================\n");

        long testStartTime = System.currentTimeMillis();
        nn.test(testData.getImages(), testData.getLabels());
        long testEndTime = System.currentTimeMillis();
        long testTime = testEndTime - testStartTime;

        System.out.println("\n========================================");
        System.out.println("  TESTE CONCLUÍDO");
        System.out.println("========================================");
        System.out.println("Tempo de teste: " + testTime + " ms (" + (testTime / 1000.0) + " segundos)");
        System.out.println("========================================\n");
    }

    private ArrayList<Layer> buildNetwork() {
        ArrayList<Layer> layers = new ArrayList<>();
        int previousSize = INPUT_SIZE;
        for (int hiddenSize : HIDDEN_LAYER_SIZES) {
            layers.add(new Layer(hiddenSize, previousSize, ActivationType.RELU));
            previousSize = hiddenSize;
        }
        layers.add(new Layer(OUTPUT_SIZE, previousSize, ActivationType.LINEAR));
        return layers;
    }

    private double[][] toOneHot(int[] labels, int numClasses) {
        double[][] targets = new double[labels.length][numClasses];
        for (int i = 0; i < labels.length; i++) {
            int label = labels[i];
            if (label >= 0 && label < numClasses) {
                targets[i][label] = 1.0;
            }
        }
        return targets;
    }

    private String architectureString() {
        StringBuilder builder = new StringBuilder();
        builder.append(INPUT_SIZE);
        for (int hidden : HIDDEN_LAYER_SIZES) {
            builder.append(" -> ").append(hidden);
        }
        builder.append(" -> ").append(OUTPUT_SIZE);
        return builder.toString();
    }
}
