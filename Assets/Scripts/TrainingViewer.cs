using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;

public class TrainingViewer : MonoBehaviour
{
    public BoxCollider2D canvasCollider;
    [SerializeField] int m_Resolution;
    Texture2D m_DrawTexture;

    ANN ann;
    int m_ImagesLoaded;
    int m_ImageHeight;
    int m_ImageWidth;
    int m_LabelsLoaded;
    byte[][] m_Images;
    byte[] m_Labels;
    double[][] m_ImageValues;
    int m_NumberOfCorrect;

    [Header("Training Parameters")]
    [SerializeField] string m_TrainingImagePath;
    [SerializeField] string m_TrainingLabelPath;
    [SerializeField] int m_Epochs;
    [SerializeField] int m_MiniBatchSize;

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
        CreateCanvas();
        SetupImages();
        StartCoroutine(StartTraining());
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    IEnumerator StartTraining()
    {
        for (int i = 0; i < m_Epochs; i++)
        {
            int noOfIter = m_ImagesLoaded > m_LabelsLoaded ? m_LabelsLoaded : m_ImagesLoaded;
            for (int j = 0; j < noOfIter; j++)
            {
                List<double> predicted = ann.Train(BlackWhiteImage(m_ImageValues[j], 0.5f).ToList(), LabelToOutputValue(m_Labels[j]).ConvertAll(x => (double)x));
                //Debug.Log($"Predicted = {PrintList(predicted)}\nExpected = {PrintList(LabelToOutputValue(m_Labels[j]))}");
                //Debug.Log($"Predicted: {OutputToLabelValue(predicted)} Actual: {m_Labels[j]}");
                if(OutputToLabelValue(predicted) == m_Labels[j])
                {
                    m_NumberOfCorrect++;
                }
                if(j % m_MiniBatchSize == 0)
                {
                    m_DrawTexture.SetPixel((j + (i * noOfIter)) / m_MiniBatchSize, (m_NumberOfCorrect * 100) / m_MiniBatchSize, Color.green);
                    m_DrawTexture.Apply();
                    Debug.Log($"{j} {m_MiniBatchSize} {(j + (i * noOfIter)) / m_MiniBatchSize}");
                    m_NumberOfCorrect = 0;
                    yield return null;
                }
            }
        }
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
    double[] BlackWhiteImage(double[] image, float threshold)
    {
        double[] newImage = new double[image.Length];

        for (int i = 0; i < image.Length; i++)
        {
            if (image[i] >= threshold)
            {
                newImage[i] = 1;
            }
            else
            {
                newImage[i] = 0;
            }
        }

        return newImage;
    }
    List<byte> LabelToOutputValue(byte value)
    {
        List<byte> output = new();
        for (int i = 0; i < 10; i++)
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
        for (int i = 0; i < value.Count; i++)
        {
            if (value[i] > output)
            {
                output = value[i];
                idx = i;
            }
        }

        return idx;
    }
    void CreateCanvas()
    {
        m_DrawTexture = new Texture2D(m_Resolution * 2, m_Resolution);
        m_DrawTexture.filterMode = FilterMode.Point;
        m_DrawTexture.wrapMode = TextureWrapMode.Clamp;

        canvasCollider.GetComponent<MeshRenderer>().material.mainTexture = m_DrawTexture;
        for (int i = 0; i < m_DrawTexture.width; i++)
        {
            for (int j = 0; j < m_DrawTexture.height; j++)
            {
                m_DrawTexture.SetPixel(i, j, Color.black);
            }
        }
        m_DrawTexture.Apply();
    }
}
