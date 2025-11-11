import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

public class Neuron {

    private final ArrayList<Double> weights;
    private final double[] weightGradients;
    private double bias;
    private double biasGradient;
    private double output;
    private double delta;
    private double netInput;
    private final ActivationType activationType;

    public Neuron(int inputSize, ActivationType activationType) {
        this.activationType = activationType;
        this.weights = new ArrayList<>(inputSize);
        double initRange = activationType == ActivationType.RELU ? Math.sqrt(2.0 / inputSize) : Math.sqrt(1.0 / inputSize);
        ThreadLocalRandom random = ThreadLocalRandom.current();

        for (int i = 0; i < inputSize; i++) {
            double value = random.nextDouble(-initRange, initRange);
            weights.add(value);
        }
        this.bias = 0.0;
        this.weightGradients = new double[inputSize];
        this.biasGradient = 0.0;
    }

    public double netInput(double[] inputs) {
        double sum = bias;
        for (int i = 0; i < weights.size(); i++) {
            sum += weights.get(i) * inputs[i];
        }
        this.netInput = sum;
        return sum;
    }

    public double activate(double[] inputs) {
        double z = netInput(inputs);
        switch (activationType) {
            case RELU:
                output = Math.max(0, z);
                break;
            case SIGMOID:
                output = 1.0 / (1.0 + Math.exp(-z));
                break;
            case LINEAR:
            default:
                output = z;
        }
        return output;
    }

    public void accumulateGradients(double[] inputs) {
        for (int j = 0; j < weights.size(); j++) {
            weightGradients[j] += delta * inputs[j];
        }
        biasGradient += delta;
    }

    public void applyGradients(double learningRate, int batchSize) {
        double scale = learningRate / batchSize;
        for (int j = 0; j < weights.size(); j++) {
            double updated = weights.get(j) - scale * weightGradients[j];
            weights.set(j, updated);
            weightGradients[j] = 0.0;
        }
        bias -= scale * biasGradient;
        biasGradient = 0.0;
    }

    public void resetGradients() {
        for (int j = 0; j < weightGradients.length; j++) {
            weightGradients[j] = 0.0;
        }
        biasGradient = 0.0;
    }

    public double activationDerivative() {
        switch (activationType) {
            case RELU:
                return netInput > 0 ? 1.0 : 0.0;
            case SIGMOID:
                return output * (1.0 - output);
            case LINEAR:
            default:
                return 1.0;
        }
    }

    public ArrayList<Double> getWeights() {
        return weights;
    }

    public void setWeights(ArrayList<Double> weights) {
        for (int i = 0; i < weights.size(); i++) {
            this.weights.set(i, weights.get(i));
        }
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getOutput() {
        return output;
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    public ActivationType getActivationType() {
        return activationType;
    }
}
