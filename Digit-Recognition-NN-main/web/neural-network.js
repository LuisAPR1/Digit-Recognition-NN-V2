const ActivationType = {
    RELU: 'relu',
    SIGMOID: 'sigmoid',
    LINEAR: 'linear'
};

class Neuron {
    constructor(inputSize, activationType) {
        this.activationType = activationType;
        this.weights = [];
        const initRange = activationType === ActivationType.RELU ? Math.sqrt(2 / inputSize) : Math.sqrt(1 / inputSize);
        for (let i = 0; i < inputSize; i++) {
            this.weights.push((Math.random() * 2 - 1) * initRange);
        }
        this.bias = 0;
        this.output = 0;
    }

    netInput(inputs) {
        let sum = this.bias;
        for (let i = 0; i < this.weights.length; i++) {
            sum += this.weights[i] * inputs[i];
        }
        return sum;
    }

    activate(inputs) {
        const z = this.netInput(inputs);
        switch (this.activationType) {
            case ActivationType.RELU:
                this.output = Math.max(0, z);
                break;
            case ActivationType.SIGMOID:
                this.output = 1 / (1 + Math.exp(-z));
                break;
            case ActivationType.LINEAR:
            default:
                this.output = z;
        }
        return this.output;
    }
}

class Layer {
    constructor(numNeurons, inputSize, activationType) {
        this.activationType = activationType;
        this.neurons = [];
        for (let i = 0; i < numNeurons; i++) {
            this.neurons.push(new Neuron(inputSize, activationType));
        }
    }

    forward(inputs) {
        const outputs = [];
        for (let i = 0; i < this.neurons.length; i++) {
            outputs.push(this.neurons[i].activate(inputs));
        }
        return outputs;
    }
}

class NeuralNetwork {
    constructor(layers) {
        this.layers = layers;
    }

    forward(input) {
        let outputs = input;
        for (let layer of this.layers) {
            outputs = layer.forward(outputs);
        }
        return outputs;
    }

    static softmax(logits) {
        const max = Math.max(...logits);
        const exps = logits.map(v => Math.exp(v - max));
        const sum = exps.reduce((acc, v) => acc + v, 0);
        return exps.map(v => v / sum);
    }

    loadWeightsFromJSON(weightsData) {
        let weightIndex = 0;

        for (let layer of this.layers) {
            for (let neuron of layer.neurons) {
                const weights = [];
                const numWeights = neuron.weights.length;

                for (let i = 0; i < numWeights; i++) {
                    weights.push(weightsData[weightIndex]);
                    weightIndex++;
                }

                const bias = weightsData[weightIndex];
                weightIndex++;

                neuron.weights = weights;
                neuron.bias = bias;
            }
        }
    }

    loadWeightsFromCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const weightsData = [];

        for (let line of lines) {
            if (line.trim() === '') continue;
            const tokens = line.split(',');
            for (let token of tokens) {
                const trimmed = token.trim();
                if (trimmed === '') continue;
                const value = parseFloat(trimmed);
                if (!isNaN(value)) {
                    weightsData.push(value);
                }
            }
        }

        const expectedValues = this.layers.reduce((total, layer) => {
            return total + layer.neurons.reduce((acc, neuron) => acc + neuron.weights.length + 1, 0);
        }, 0);

        if (weightsData.length !== expectedValues) {
            throw new Error(`Ficheiro de pesos incompatível. Esperado ${expectedValues} valores, recebido ${weightsData.length}.`);
        }

        this.loadWeightsFromJSON(weightsData);
    }

    predict(inputs) {
        const logits = this.forward(inputs);
        return NeuralNetwork.softmax(logits);
    }
}
