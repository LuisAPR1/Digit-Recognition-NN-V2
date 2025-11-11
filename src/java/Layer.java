import java.util.ArrayList;

public class Layer {
    private final ArrayList<Neuron> neurons;
    private final ActivationType activationType;

    public Layer(int numNeurons, int inputSize, ActivationType activationType) {
        this.activationType = activationType;
        neurons = new ArrayList<>();
        for (int i = 0; i < numNeurons; i++) {
            neurons.add(new Neuron(inputSize, activationType));
        }
    }

    public double[] forward(double[] inputs) {
        double[] outputs = new double[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            outputs[i] = neurons.get(i).activate(inputs);
        }
        return outputs;
    }

    public void accumulateGradients(double[] inputs) {
        for (Neuron neuron : neurons) {
            neuron.accumulateGradients(inputs);
        }
    }

    public void applyGradients(double learningRate, int batchSize) {
        for (Neuron neuron : neurons) {
            neuron.applyGradients(learningRate, batchSize);
        }
    }

    public void resetGradients() {
        for (Neuron neuron : neurons) {
            neuron.resetGradients();
        }
    }

    public ArrayList<Neuron> getNeurons() {
        return neurons;
    }

    public ActivationType getActivationType() {
        return activationType;
    }
}
