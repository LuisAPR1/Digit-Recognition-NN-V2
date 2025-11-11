import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNetwork {
    private final ArrayList<Layer> layers;
    private final double[][] lastInputsPerLayer;
    private final double[][] lastOutputsPerLayer;
    private final boolean useSoftmaxOutput = true;
    private final int batchSize = 64;
    private double learningRate;

    public NeuralNetwork(ArrayList<Layer> layers) {
        this.layers = layers;
        this.lastInputsPerLayer = new double[layers.size()][];
        this.lastOutputsPerLayer = new double[layers.size()][];
    }

    public void train(double[][] inputs, double[][] targets, double lossThreshold, double learningRate,
                      String lossLogPath, String weightsPath) {
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException("Inputs e targets têm comprimentos diferentes.");
        }

        this.learningRate = learningRate;
        int epoch = 0;
        List<Double> lossHistory = new ArrayList<>();
        Double previousEpochLoss = null;

        try {
            ensureParentDirectory(lossLogPath);
            ensureParentDirectory(weightsPath);
        } catch (IOException e) {
            throw new RuntimeException("Não foi possível preparar diretórios de saída: " + e.getMessage(), e);
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(lossLogPath))) {

            while (true) {
                shuffleData(inputs, targets);
                double totalLoss = 0.0;
                int samplesProcessed = 0;
                int progressInterval = Math.max(1, inputs.length / 20);

                for (int start = 0; start < inputs.length; start += batchSize) {
                    int end = Math.min(start + batchSize, inputs.length);
                    resetGradients();

                    for (int i = start; i < end; i++) {
                        double[] output = forward(inputs[i]);
                        totalLoss += sampleError(output, targets[i]);
                        backward(targets[i]);
                        accumulateGradients();

                        samplesProcessed++;
                        if (samplesProcessed % progressInterval == 0 || samplesProcessed == inputs.length) {
                            double partialLoss = totalLoss / samplesProcessed;
                            System.out.println(String.format("   -> Época %d | Amostra %d/%d | Loss parcial: %.6f",
                                    epoch + 1, samplesProcessed, inputs.length, partialLoss));
                        }
                    }

                    applyGradients(end - start);
                }

                double epochLoss = totalLoss / inputs.length;
                lossHistory.add(epochLoss);

                writer.write(" " + String.format("%.100f", epochLoss).replace('.', ','));
                writer.newLine();

                epoch++;

                double delta = previousEpochLoss == null ? 0.0 : epochLoss - previousEpochLoss;
                String deltaStr = previousEpochLoss == null ? "" : String.format(" (Δ %.6f)", delta);
                System.out.println("Época " + epoch + " - Loss: " + String.format("%.6f", epochLoss) + deltaStr);
                previousEpochLoss = epochLoss;

                if (epochLoss < lossThreshold) {
                    System.out.println("\n✓ Treinamento convergiu na época: " + epoch);
                    System.out.println("  Loss final: " + String.format("%.6f", epochLoss));
                    break;
                }

                if (epoch % 10 == 0 && epoch >= 10) {
                    double previousLoss = lossHistory.get(lossHistory.size() - 10);
                    if (epochLoss > previousLoss) {
                        System.out.println("O loss aumentou na época: " + epoch + ". Interrompendo o treinamento.");
                        break;
                    }
                }
            }

            saveWeights(weightsPath);

        } catch (IOException e) {
            System.err.println("Erro ao escrever valores de loss no arquivo: " + e.getMessage());
        }
    }

    public double[] forward(double[] input) {
        double[] current = input;
        for (int i = 0; i < layers.size(); i++) {
            lastInputsPerLayer[i] = current;
            current = layers.get(i).forward(current);
            lastOutputsPerLayer[i] = current;
        }

        if (useSoftmaxOutput) {
            double[] probabilities = softmax(current);
            lastOutputsPerLayer[lastOutputsPerLayer.length - 1] = probabilities;
            return probabilities;
        }
        return current;
    }

    private double[] softmax(double[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (double logit : logits) {
            if (logit > max) {
                max = logit;
            }
        }
        double sum = 0.0;
        double[] expValues = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            expValues[i] = Math.exp(logits[i] - max);
            sum += expValues[i];
        }
        for (int i = 0; i < expValues.length; i++) {
            expValues[i] /= sum;
        }
        return expValues;
    }

    public void backward(double[] target) {
        Layer outputLayer = layers.get(layers.size() - 1);
        double[] outputProbabilities = lastOutputsPerLayer[lastOutputsPerLayer.length - 1];

        for (int i = 0; i < outputLayer.getNeurons().size(); i++) {
            Neuron neuron = outputLayer.getNeurons().get(i);
            double delta = outputProbabilities[i] - target[i];
            neuron.setDelta(delta);
        }

        for (int i = layers.size() - 2; i >= 0; i--) {
            Layer currentLayer = layers.get(i);
            Layer nextLayer = layers.get(i + 1);
            for (int j = 0; j < currentLayer.getNeurons().size(); j++) {
                Neuron neuron = currentLayer.getNeurons().get(j);
                double sum = 0.0;
                for (Neuron nextNeuron : nextLayer.getNeurons()) {
                    sum += nextNeuron.getWeights().get(j) * nextNeuron.getDelta();
                }
                neuron.setDelta(sum * neuron.activationDerivative());
            }
        }
    }

    private void accumulateGradients() {
        for (int i = 0; i < layers.size(); i++) {
            double[] inputs = lastInputsPerLayer[i];
            layers.get(i).accumulateGradients(inputs);
        }
    }

    private void applyGradients(int batchSamples) {
        for (Layer layer : layers) {
            layer.applyGradients(learningRate, batchSamples);
        }
    }

    private void resetGradients() {
        for (Layer layer : layers) {
            layer.resetGradients();
        }
    }

    public void test(double[][] inputs, int[] labels) {
        double totalLoss = 0.0;
        int correct = 0;

        for (int i = 0; i < inputs.length; i++) {
            double[] prediction = forward(inputs[i]);
            int predictedLabel = argMax(prediction);
            if (predictedLabel == labels[i]) {
                correct++;
            }
            totalLoss += sampleError(prediction, oneHot(labels[i]));
        }

        double averageLoss = totalLoss / inputs.length;
        double accuracy = (double) correct / inputs.length * 100.0;

        System.out.println("\n════════════════════════════════════════");
        System.out.println("  RESULTADOS DO TESTE");
        System.out.println("════════════════════════════════════════");
        System.out.println("Accuracy: " + String.format("%.2f", accuracy) + "%");
        System.out.println("Corretos: " + correct + " de " + inputs.length);
        System.out.println("Loss médio: " + String.format("%.6f", averageLoss));
        System.out.println("════════════════════════════════════════");
    }

    public double[][] predict(double[][] inputs) {
        double[][] outputs = new double[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            outputs[i] = forward(inputs[i]);
        }
        return outputs;
    }

    private double sampleError(double[] output, double[] target) {
        double error = 0.0;
        for (int i = 0; i < output.length; i++) {
            double eps = 1e-9;
            error -= target[i] * Math.log(output[i] + eps);
        }
        return error;
    }

    private int argMax(double[] values) {
        int index = 0;
        double best = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > best) {
                best = values[i];
                index = i;
            }
        }
        return index;
    }

    private double[] oneHot(int label) {
        double[] target = new double[getOutputSize()];
        if (label >= 0 && label < target.length) {
            target[label] = 1.0;
        }
        return target;
    }

    private int getOutputSize() {
        return layers.get(layers.size() - 1).getNeurons().size();
    }

    private void shuffleData(double[][] inputs, double[][] targets) {
        Random random = new Random();
        for (int i = inputs.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            double[] tempInput = inputs[i];
            inputs[i] = inputs[j];
            inputs[j] = tempInput;

            double[] tempTarget = targets[i];
            targets[i] = targets[j];
            targets[j] = tempTarget;
        }
    }

    public void saveWeights(String filename) {
        try {
            ensureParentDirectory(filename);
        } catch (IOException e) {
            throw new RuntimeException("Não foi possível criar diretório para " + filename + ": " + e.getMessage(), e);
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            for (Layer layer : layers) {
                for (Neuron neuron : layer.getNeurons()) {
                    for (double weight : neuron.getWeights()) {
                        writer.write(weight + ",");
                    }
                    writer.write(neuron.getBias() + "");
                    writer.newLine();
                }
            }
            System.out.println("Pesos salvos em " + filename);
        } catch (IOException e) {
            System.err.println("Erro ao salvar os pesos: " + e.getMessage());
        }
    }

    public void loadWeights(String filename) {
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            int layerIndex = 0;
            int neuronIndex = 0;
            int lineNumber = 0;

            while ((line = reader.readLine()) != null) {
                lineNumber++;

                String[] tokens = line.split(",");
                Layer layer = layers.get(layerIndex);
                Neuron neuron = layer.getNeurons().get(neuronIndex);
                int expectedWeights = neuron.getWeights().size();
                int providedWeights = tokens.length - 1;

                if (providedWeights != expectedWeights) {
                    throw new IOException(String.format(
                            "Linha %d inválida em %s: esperado %d pesos + 1 bias, mas recebido %d valores.",
                            lineNumber, filename, expectedWeights, providedWeights));
                }

                ArrayList<Double> weights = new ArrayList<>();
                for (int i = 0; i < providedWeights; i++) {
                    weights.add(Double.parseDouble(tokens[i]));
                }
                double bias = Double.parseDouble(tokens[tokens.length - 1]);

                neuron.setWeights(weights);
                neuron.setBias(bias);

                neuronIndex++;
                if (neuronIndex >= layer.getNeurons().size()) {
                    neuronIndex = 0;
                    layerIndex++;
                    if (layerIndex >= layers.size()) {
                        break;
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("Erro ao carregar os pesos: " + e.getMessage());
        }
    }

    private void ensureParentDirectory(String filePath) throws IOException {
        Path parent = Path.of(filePath).getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
    }
}
