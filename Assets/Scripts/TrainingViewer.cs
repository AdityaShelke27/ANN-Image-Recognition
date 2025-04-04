using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;

public class TrainingViewer : MonoBehaviour
{
    public BoxCollider2D canvasCollider;
    [SerializeField] int m_Resolution;

    ANN ann;
    int m_TrainingImagesLoaded;
    int m_TrainingImageHeight;
    int m_TrainingImageWidth;
    int m_TrainingLabelsLoaded;
    byte[][] m_TrainingImages;
    byte[] m_TrainingLabels;
    double[][] m_TrainingImageValues;
    int m_NumberOfCorrectTraining;
    [SerializeField] LineRenderer m_LineRendererTraining;

    int m_TestingImagesLoaded;
    int m_TestingImageHeight;
    int m_TestingImageWidth;
    int m_TestingLabelsLoaded;
    byte[][] m_TestingImages;
    byte[] m_TestingLabels;
    double[][] m_TestingImageValues;
    int m_NumberOfCorrectTesting;
    [SerializeField] LineRenderer m_LineRendererTesting;

    [Header("Training Parameters")]
    [SerializeField] string m_TrainingImagePath;
    [SerializeField] string m_TrainingLabelPath;
    [SerializeField] string m_TestingImagePath;
    [SerializeField] string m_TestingLabelPath;
    [SerializeField] int m_Epochs;
    [SerializeField] int m_MiniBatchSize;
    [SerializeField] float m_BlackWhiteThreshold;

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

        SetupTrainingImages();
        SetupTestingImages();
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
            int noOfIter = m_TrainingImagesLoaded > m_TrainingLabelsLoaded ? m_TrainingLabelsLoaded : m_TrainingImagesLoaded;
            for (int j = 0; j < noOfIter; j++)
            {
                List<double> predicted = ann.Train(BlackWhiteImage(m_TrainingImageValues[j], m_BlackWhiteThreshold).ToList(), LabelToOutputValue(m_TrainingLabels[j]).ConvertAll(x => (double)x));
                //Debug.Log($"Predicted = {PrintList(predicted)}\nExpected = {PrintList(LabelToOutputValue(m_Labels[j]))}");
                //Debug.Log($"Predicted: {OutputToLabelValue(predicted)} Actual: {m_Labels[j]}");
                if(OutputToLabelValue(predicted) == m_TrainingLabels[j])
                {
                    m_NumberOfCorrectTraining++;
                }
                if(j % m_MiniBatchSize == 0)
                {
                    m_LineRendererTraining.positionCount++;
                    m_LineRendererTraining.SetPosition((j + (i * noOfIter)) / m_MiniBatchSize, new Vector3((j + (i * noOfIter)) / m_MiniBatchSize, (m_NumberOfCorrectTraining * 100) / m_MiniBatchSize, 0));
                    Debug.Log($"{j} {m_MiniBatchSize} {(j + (i * noOfIter)) / m_MiniBatchSize}");
                    m_NumberOfCorrectTraining = 0;
                    yield return null;
                }
            }
        }

        StartCoroutine(StartTesting());
    }
    IEnumerator StartTesting()
    {
        int miniBatchTest = m_MiniBatchSize / 10;
        int noOfIter = m_TestingImagesLoaded > m_TestingLabelsLoaded ? m_TestingLabelsLoaded : m_TestingImagesLoaded;
        for (int j = 0; j < noOfIter; j++)
        {
            List<double> predicted = ann.Test(BlackWhiteImage(m_TestingImageValues[j], m_BlackWhiteThreshold).ToList());
            //Debug.Log($"Predicted = {PrintList(predicted)}\nExpected = {PrintList(LabelToOutputValue(m_Labels[j]))}");
            //Debug.Log($"Predicted: {OutputToLabelValue(predicted)} Actual: {m_Labels[j]}");
            if (OutputToLabelValue(predicted) == m_TestingLabels[j])
            {
                m_NumberOfCorrectTesting++;
            }
            if (j % miniBatchTest == 0)
            {
                m_LineRendererTesting.positionCount++;
                m_LineRendererTesting.SetPosition(j / miniBatchTest, new Vector3(j / miniBatchTest, (m_NumberOfCorrectTesting * 100) / miniBatchTest, 0));
                Debug.Log($"{j} {miniBatchTest} {j / miniBatchTest}");
                m_NumberOfCorrectTesting = 0;
                yield return null;
            }
        }
    }
    void SetupTrainingImages()
    {
        (m_TrainingImagesLoaded, m_TrainingImageHeight, m_TrainingImageWidth, m_TrainingImages) =
            DataManager.LoadImages(m_TrainingImagePath, 0);

        (m_TrainingLabelsLoaded, m_TrainingLabels) =
            DataManager.LoadLabels(m_TrainingLabelPath, 0);

        m_TrainingImageValues = new double[m_TrainingImages.Length][];
        for (int i = 0; i < m_TrainingImages.Length; i++)
        {
            m_TrainingImageValues[i] = new double[m_TrainingImages[i].Length];
            for (int j = 0; j < m_TrainingImages[i].Length; j++)
            {
                m_TrainingImageValues[i][j] = (double)m_TrainingImages[i][j] / 255;
            }
        }
    }
    void SetupTestingImages()
    {
        (m_TestingImagesLoaded, m_TestingImageHeight, m_TestingImageWidth, m_TestingImages) =
            DataManager.LoadImages(m_TestingImagePath, 0);

        (m_TestingLabelsLoaded, m_TestingLabels) =
            DataManager.LoadLabels(m_TestingLabelPath, 0);

        m_TestingImageValues = new double[m_TestingImages.Length][];
        for (int i = 0; i < m_TestingImages.Length; i++)
        {
            m_TestingImageValues[i] = new double[m_TestingImages[i].Length];
            for (int j = 0; j < m_TestingImages[i].Length; j++)
            {
                m_TestingImageValues[i][j] = (double)m_TestingImages[i][j] / 255;
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
}
