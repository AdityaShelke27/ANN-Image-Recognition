using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Collections;
using TMPro;

public class ImageClassifier : MonoBehaviour
{
    ANN ann;
    int m_ImagesLoaded;
    int m_ImageHeight;
    int m_ImageWidth;
    int m_LabelsLoaded;
    byte[][] m_Images;
    byte[] m_Labels;

    double[][] m_ImageValues;
    Texture2D texture;
    float[][] m_Kernel = new float[][] { 
        new float[] { 1, 2, 1, 0, 0, 0, -1, -2, -1 }, 
        new float[] { -1, -2, -1, 0, 0, 0, 1, 2, 1 }, 
        new float[] { 1, 0, -1, 2, 0, -2, 1, 0, -1 }, 
        new float[] { -1, 0, 1, -2, 0, 2, -1, 0, 1 },
    };
    [SerializeField] float m_BlackWhiteThreshold;
    [SerializeField] Painter m_Painter;
    [SerializeField] TMP_Text m_Text;
    [SerializeField] SwitchImages m_SwitchImages;

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
        texture = new Texture2D(28, 28, TextureFormat.RGBA32, false);
        //RenderTexture.active = m_Painter.m_DrawTexture;
        ann = new ANN(m_NoOfInputs, m_NoOfOutputs, m_NoOfHiddenLayers, m_NoOfNeuronsPerHiddenLayers,
            m_LearningRate, m_HiddenActivation, m_OutputActivation);
        SetupImages();
        StartTraining();
        
        StartCoroutine(PredictCanvas());
        Debug.Log("Done");
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    IEnumerator PredictCanvas()
    {
        yield return null;
        RenderTexture.active = m_Painter.m_DrawTexture;
        texture.ReadPixels(new Rect(0, 0, m_Painter.m_DrawTexture.width, m_Painter.m_DrawTexture.height), 0, 0);
        texture.Apply();
        RenderTexture.active = null;
        List<double> pixels = new();
        Color[] colArr = texture.GetPixels(0, 0, texture.width, texture.height);
        for (int i = 0; i < colArr.Length; i++)
        {
            pixels.Add((double)colArr[i].r);
        }
        List<double> predicted = ann.Test(pixels);

        m_Text.text = OutputToLabelValue(predicted).ToString();
        //Debug.Log(OutputToLabelValue(predicted));
        StartCoroutine(PredictCanvas());
    }
    void SetupImages()
    {
        (m_ImagesLoaded, m_ImageHeight, m_ImageWidth, m_Images) =
            DataManager.LoadImages(m_TrainingImagePath, 0);

        (m_LabelsLoaded, m_Labels) =
            DataManager.LoadLabels(m_TrainingLabelPath, 0);

        m_ImageValues = new double[m_Images.Length][];
        for (int i = 0; i < m_Images.Length; i++)
        {
            m_ImageValues[i] = new double[m_Images[i].Length];
            for (int j = 0; j < m_Images[i].Length; j++)
            {
                m_ImageValues[i][j] = (double)m_Images[i][j] / 255;
            }
        }
    }
    void StartTraining()
    {
        int noOfIter = m_ImagesLoaded > m_LabelsLoaded ? m_LabelsLoaded : m_ImagesLoaded;
        noOfIter = noOfIter > m_NoOfDataPointsToTrain ? m_NoOfDataPointsToTrain : noOfIter;
        for (int i = 0; i < m_Epochs; i++)
        {
            for (int j = 0; j < noOfIter; j++)
            {
                List<double> predicted = ann.Train(ImageProcessor.BlackWhiteImage(m_ImageValues[j], m_BlackWhiteThreshold).ToList(), LabelToOutputValue(m_Labels[j]).ConvertAll(x => (double)x));
                //Debug.Log($"Predicted = {PrintList(predicted)}\nExpected = {PrintList(LabelToOutputValue(m_Labels[j]))}");
                Debug.Log($"Predicted: {OutputToLabelValue(predicted)} Actual: {m_Labels[j]}");
            }
        }
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
    int OutputToLabelValue(List<double> value)
    {
        double output = 0;
        int idx = -1;
        for(int i = 0; i < value.Count; i++)
        {
            if (value[i] > output)
            {
                output = value[i];
                idx = i;
            }
        }

        return idx;
    }
}
