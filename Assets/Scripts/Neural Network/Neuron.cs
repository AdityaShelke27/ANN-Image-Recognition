using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Neuron
{
    public int numInputs;
    public double nWeights;
    public List<double> weights = new List<double>();
    public List<double> inputs = new List<double>();
    public double bias;
    public double output;
    public double errorGradient;

    public List<double> weightGradients;
    public double biasGradient;

    public Neuron(int nInputs)
    {
        numInputs = nInputs;
        nWeights = numInputs;

        bias = Random.Range(-1f, 1f);

        for (int i = 0; i < nWeights; i++)
        {
            float limit = Mathf.Sqrt(2.0f / numInputs);
            float weight = Random.Range(-limit, limit);
            //weights.Add(Random.Range(-1f, 1f));
            weights.Add(weight);
        }

        InitializeGradients();
    }

    public void InitializeGradients()
    {
        weightGradients = new List<double>(new double[numInputs]);
        biasGradient = 0.0;
    }
}