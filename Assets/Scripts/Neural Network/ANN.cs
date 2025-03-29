using System;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

public enum Activation
{
    Sigmoid,
    Step,
    Tanh,
    ReLu,
    LeakyReLu,
    SoftMax
}
public class ANN
{
    public int numInputs;
    public int numOutputs;
    public int numHidden;
    public int numNPerHidden;
    Activation hiddenActivation;
    Activation outputActivation;
    public double alpha;
    public List<Layer> layers = new List<Layer>();

    public ANN(int nI, int nO, int nH, int nPH, double a, Activation hiddenActivation, Activation outputActivation)
    {
        numInputs = nI;
        numOutputs = nO;
        numHidden = nH;
        numNPerHidden = nPH;
        alpha = a;

        if (numHidden > 0)
        {
            layers.Add(new Layer(numInputs, numNPerHidden));

            for (int i = 0; i < numHidden - 1; i++)
            {
                layers.Add(new Layer(numNPerHidden, numNPerHidden));
            }

            layers.Add(new Layer(numNPerHidden, numOutputs));
        }
        else
        {
            layers.Add(new Layer(numInputs, numOutputs));
        }

        this.hiddenActivation = hiddenActivation;
        this.outputActivation = outputActivation;
    }

    public List<double> Train(List<double> inputValues, List<double> desiredOutput)
    {
        List<double> output = Test(inputValues);
        UpdateWeights(output, desiredOutput);

        return output;
    }

    public List<double> Test(List<double> inputValues)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();

        if (inputValues.Count != numInputs)
        {
            Debug.Log("ERROR: Number of Inputs must be " + numInputs);
            return outputs;
        }

        inputs = new List<double>(inputValues);

        for (int i = 0; i < numHidden + 1; i++)
        {
            if (i > 0)
            {
                inputs = new List<double>(outputs);
            }
            outputs.Clear();

            List<double> weightedInputs = new();
            for (int j = 0; j < layers[i].numNeurons; j++)
            {
                double N = 0;
                layers[i].neurons[j].inputs.Clear();

                for (int k = 0; k < layers[i].neurons[j].numInputs; k++)
                {
                    layers[i].neurons[j].inputs.Add(inputs[k]);
                    N += inputs[k] * layers[i].neurons[j].weights[k];
                }

                N -= layers[i].neurons[j].bias;
                weightedInputs.Add(N);
            }
            for(int j = 0; j < layers[i].numNeurons; j++)
            {
                if (i == numHidden)
                {
                    layers[i].neurons[j].output = ActivationO(weightedInputs, j);
                }
                else
                {
                    layers[i].neurons[j].output = Activation(weightedInputs, j);
                }
                outputs.Add(layers[i].neurons[j].output);
            }
        }

        return outputs;
    }

    void UpdateWeights(List<double> outputs, List<double> desiredOutputs)
    {
        double error;

        for (int i = numHidden; i >= 0; i--)
        {
            for (int j = 0; j < layers[i].numNeurons; j++)
            {
                if (i == numHidden)
                {

                    error = desiredOutputs[j] - outputs[j];
                    layers[i].neurons[j].errorGradient = outputs[j] * (1 - outputs[j]) * error;
                }
                else
                {
                    double sumError = 0;
                    for (int k = 0; k < layers[i + 1].numNeurons; k++)
                    {
                        sumError += layers[i + 1].neurons[k].errorGradient * layers[i + 1].neurons[k].weights[j];
                    }

                    layers[i].neurons[j].errorGradient = layers[i].neurons[j].output * (1 - layers[i].neurons[j].output) * sumError;
                }

                for (int k = 0; k < layers[i].neurons[j].numInputs; k++)
                {
                    if (i == numHidden)
                    {
                        error = desiredOutputs[j] - outputs[j];
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * error;
                    }
                    else
                    {
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * layers[i].neurons[j].errorGradient;
                    }
                }

                layers[i].neurons[j].bias -= alpha * layers[i].neurons[j].errorGradient;
            }
        }
    }

    public void CopyWeights(ANN parent1)
    {
        double sum = 0;
        double sum2 = 0;
        double sum3 = 0;
        for (int i = 0; i < parent1.layers.Count; i++)
        {
            for (int j = 0; j < parent1.layers[i].neurons.Count; j++)
            {
                for (int k = 0; k < parent1.layers[i].neurons[j].weights.Count; k++)
                {
                    sum2 += layers[i].neurons[j].weights[k];
                    layers[i].neurons[j].weights[k] = parent1.layers[i].neurons[j].weights[k];
                    sum += parent1.layers[i].neurons[j].weights[k];
                    sum3 += layers[i].neurons[j].weights[k];
                }
                sum2 += layers[i].neurons[j].bias;
                layers[i].neurons[j].bias = parent1.layers[i].neurons[j].bias;
                sum += parent1.layers[i].neurons[j].bias;
                sum3 += layers[i].neurons[j].bias;
            }
        }

        //Debug.Log(sum + " " + sum2 + " " + sum3 + " ************************");
    }
    public void WeightSum()
    {
        double sum = 0;
        for (int i = 0; i < layers.Count; i++)
        {
            for (int j = 0; j < layers[i].neurons.Count; j++)
            {
                for (int k = 0; k < layers[i].neurons[j].weights.Count; k++)
                {
                    layers[i].neurons[j].weights[k] = layers[i].neurons[j].weights[k];
                    sum += layers[i].neurons[j].weights[k];
                }
                sum += layers[i].neurons[j].bias;
            }
        }
    }
    public void CopyWeights(ANN parent1, float littleMutation)
    {
        for (int i = 0; i < parent1.layers.Count; i++)
        {
            for (int j = 0; j < parent1.layers[i].neurons.Count; j++)
            {
                for (int k = 0; k < parent1.layers[i].neurons[j].weights.Count; k++)
                {
                    layers[i].neurons[j].weights[k] = parent1.layers[i].neurons[j].weights[k] + Random.Range(-littleMutation, littleMutation);
                }

                layers[i].neurons[j].bias = parent1.layers[i].neurons[j].bias + Random.Range(-littleMutation, littleMutation);
            }
        }
    }
    public string PrintWeights()
    {
        string weightStr = "";
        foreach (Layer l in layers)
        {
            foreach (Neuron n in l.neurons)
            {
                foreach (double w in n.weights)
                {
                    weightStr += w + ",";
                }
                weightStr += n.bias + ",";
            }
        }
        return weightStr;
    }
    public void LoadWeights(string weightStr)
    {
        if (weightStr == "") return;
        string[] weightValues = weightStr.Split(',');
        int w = 0;
        foreach (Layer l in layers)
        {
            foreach (Neuron n in l.neurons)
            {
                for (int i = 0; i < n.weights.Count; i++)
                {
                    n.weights[i] = System.Convert.ToDouble(weightValues[w]);
                    w++;
                }
                n.bias = System.Convert.ToDouble(weightValues[w]);
                w++;
            }
        }
    }
    double Activation(List<double> inputs, int idx)
    {
        switch (hiddenActivation)
        {
            case global::Activation.Sigmoid:
                return Sigmoid(inputs[idx]);
            case global::Activation.Tanh:
                return Tanh(inputs[idx]);
            case global::Activation.ReLu:
                return ReLu(inputs[idx]);
            case global::Activation.Step:
                return Step(inputs[idx]);
            case global::Activation.LeakyReLu:
                return LeakyReLu(inputs[idx]);
            case global::Activation.SoftMax:
                return SoftMax(inputs, idx);
            default:
                return Sigmoid(inputs[idx]);
        }
    }
    double ActivationO(List<double> inputs, int idx)
    {
        switch (outputActivation)
        {
            case global::Activation.Sigmoid:
                return Sigmoid(inputs[idx]);
            case global::Activation.Tanh:
                return Tanh(inputs[idx]);
            case global::Activation.ReLu:
                return ReLu(inputs[idx]);
            case global::Activation.Step:
                return Step(inputs[idx]);
            case global::Activation.LeakyReLu:
                return LeakyReLu(inputs[idx]);
            case global::Activation.SoftMax:
                return SoftMax(inputs, idx);
            default:
                return Sigmoid(inputs[idx]);
        }
    }
    double Sigmoid(double input)
    {
        input = System.Math.Clamp(input, -1, 1);
        double expo = System.Math.Exp(input);
        double result = expo / (1 + expo);
        //if (result == double.NaN)
        //{
        //Debug.Log(result + " " + expo + " " + input);
        //}
        result = System.Math.Clamp(result, -1, 1);
        return result;
    }
    double Step(double input)
    {
        if (input > 0)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }

    double Tanh(double input)
    {
        double expo = Math.Exp(-2 * input);
        return 2 / (1f + expo) - 1;
    }

    double ReLu(double input)
    {
        if (input > 0)
        {
            return input;
        }
        else
        {
            return 0;
        }
    }
    double LeakyReLu(double input)
    {
        if (input > 0)
        {
            return input;
        }
        else
        {
            return 0.01 * input;
        }
    }
    double SoftMax(List<double> input, int idx)
    {
        double sum = 0;
        foreach (double t in input)
        {
            sum += Math.Exp(t);
        }

        return Math.Exp(input[idx]) / sum;
    }
}