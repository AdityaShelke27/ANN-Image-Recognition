using System.Linq;
using System.Collections.Generic;
using UnityEngine;

public class ImageClassifier : MonoBehaviour
{
    ANN ann;
    int m_ImagesLoaded;
    int m_ImageHeight;
    int m_ImageWidth;
    int m_LabelsLoaded;
    byte[][] m_Images;
    byte[] m_Labels;

    [Header("Training Parameters")]
    [SerializeField] string m_TrainingImagePath;
    [SerializeField] string m_TrainingLabelPath;
    [SerializeField] int m_Epochs;
    [SerializeField] int m_NoOfDataPointsToTrain;

    [Header("ANN Parameters")]
    [SerializeField] int m_NoOfInputs;
    [SerializeField] int m_NoOfOutputs;
    [SerializeField] int m_NoOfHiddenLayers;
    [SerializeField] int m_NoOfNeuronsPerHiddenLayers;
    [SerializeField] float m_LearningRate;
    [SerializeField] Activation m_HiddenActivation;
    [SerializeField] Activation m_OutputActivation;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        ann = new ANN(m_NoOfInputs, m_NoOfOutputs, m_NoOfHiddenLayers, m_NoOfNeuronsPerHiddenLayers, 
            m_LearningRate, m_HiddenActivation, m_OutputActivation);

        (m_ImagesLoaded, m_ImageHeight, m_ImageWidth, m_Images) = 
            DataManager.LoadImages(m_TrainingImagePath, 0);

        (m_LabelsLoaded, m_Labels) =
            DataManager.LoadLabels(m_TrainingLabelPath, 0);

        int noOfIter = m_ImagesLoaded > m_LabelsLoaded ? m_LabelsLoaded : m_ImagesLoaded;
        noOfIter = noOfIter > m_NoOfDataPointsToTrain ? m_NoOfDataPointsToTrain : noOfIter;
        for(int i = 0; i < m_Epochs; i++)
        {
            for(int j = 0; j < noOfIter; j++)
            {
                List<double> predicted = ann.Train(m_Images[j].ToList().ConvertAll(x => (double)x), LabelToOutputValue(m_Labels[j]).ConvertAll(x => (double)x));
                Debug.Log($"Predicted = {PrintList(predicted)}\nExpected = {PrintList(LabelToOutputValue(m_Labels[j]))}");
            }
        }
        Debug.Log("Done");
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    string PrintList<T>(List<T> list)
    {
        string str = "{ ";
        foreach (T t in list)
        {
            str += t.ToString() + ", ";
        }
        str += "}";

        return str;
    }
    List<byte> LabelToOutputValue(byte value)
    {
        List<byte> output = new();
        for(int i = 0; i < 10; i++)
        {
            if (i != value)
            {
                output.Add(0);
            }
            else
            {
                output.Add(1);
            }
        }

        return output;
    }
}
