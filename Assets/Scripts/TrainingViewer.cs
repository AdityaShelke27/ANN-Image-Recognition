using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UIElements;
using Random = UnityEngine.Random;

public class TrainingViewer : MonoBehaviour
{
    public BoxCollider2D canvasCollider;
    [SerializeField] int m_Resolution;
    [SerializeField] int m_DataPerPoint;

    [SerializeField] RenderTexture m_ImageTexture;

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

    [SerializeField] LineRenderer m_LineRendererLoss;

    [Header("Training Parameters")]
    [SerializeField] string m_TrainingImagePath;
    [SerializeField] string m_TrainingLabelPath;
    [SerializeField] string m_TestingImagePath;
    [SerializeField] string m_TestingLabelPath;
    [SerializeField] int m_Epochs;
    [SerializeField] int m_MiniBatchSize;
    [SerializeField] float m_BlackWhiteThreshold;
    [SerializeField] Vector2 m_RotationRandomizer;
    [SerializeField] Vector2 m_PositionRandomizer;
    [SerializeField] Vector2 m_ScaleRandomizer;
    double[][] m_Kernel = new double[][] {
        new double[] { 1, 2, 1, 0, 0, 0, -1, -2, -1 },
        new double[] { -1, -2, -1, 0, 0, 0, 1, 2, 1 },
        new double[] { 1, 0, -1, 2, 0, -2, 1, 0, -1 },
        new double[] { -1, 0, 1, -2, 0, 2, -1, 0, 1 },
    };

    [Header("ANN Parameters")]
    [SerializeField] int m_NoOfInputs;
    [SerializeField] int m_NoOfOutputs;
    [SerializeField] int m_NoOfHiddenLayers;
    [SerializeField] int m_NoOfNeuronsPerHiddenLayers;
    [SerializeField] float m_LearningRate;
    [SerializeField] double m_RegularizationFactor;
    [SerializeField] Activation m_HiddenActivation;
    [SerializeField] Activation m_OutputActivation;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        ann = new ANN(m_NoOfInputs, m_NoOfOutputs, m_NoOfHiddenLayers, m_NoOfNeuronsPerHiddenLayers,
            m_LearningRate, m_HiddenActivation, m_OutputActivation, m_RegularizationFactor, m_MiniBatchSize);

        SetupTrainingImages();
        SetupTestingImages();
        StartCoroutine(StartTraining());
    }

    // Update is called once per frame
    void Update()
    {

    }

   
    RenderTexture CreateTexture()
    {
        RenderTexture rt = new RenderTexture(m_Resolution, m_Resolution, 0);
        rt.filterMode = FilterMode.Point;
        rt.wrapMode = TextureWrapMode.Clamp;
        rt.enableRandomWrite = true;

        return rt;
    }
    IEnumerator StartTraining()
    {
        for (int i = 0; i < m_Epochs; i++)
        {
            int batchCount = 0;
            int dataCount = 0;
            int noOfIter = m_TrainingImagesLoaded > m_TrainingLabelsLoaded ? m_TrainingLabelsLoaded : m_TrainingImagesLoaded;
            double lossSum = 0;
            for (int j = 0; j < noOfIter; j++)
            {
                double[] image;
                List<double> inputs = new();
                image = ImageProcessor.BlackWhiteImage(m_TrainingImageValues[j], m_BlackWhiteThreshold);
                image = ImageProcessor.TransformTexture(image, Random.Range(m_RotationRandomizer.x, m_RotationRandomizer.y), 
                        new Vector2(Random.Range(m_ScaleRandomizer.x, m_ScaleRandomizer.y), Random.Range(m_ScaleRandomizer.x, m_ScaleRandomizer.y)),
                        new Vector2(Random.Range(m_PositionRandomizer.x, m_PositionRandomizer.y), Random.Range(m_PositionRandomizer.x, m_PositionRandomizer.y)));
                for (int kels = 0; kels < m_Kernel.Length; kels++)
                {
                    double[] kernelImage;
                    kernelImage = ImageProcessor.KerneledImage(image, m_Kernel[kels]);
                    kernelImage = ImageProcessor.MaxPool(kernelImage, 2);
                    /*image = ImageProcessor.KerneledImage(image, m_Kernel[kels]);
                    image = ImageProcessor.MaxPool(image, 2);*/

                    for (int pxl = 0; pxl < kernelImage.Length; pxl++)
                    {
                        inputs.Add(kernelImage[pxl]);
                    }
                }

                List<double> predicted = ann.Train(inputs, LabelToOutputValue(m_TrainingLabels[j]).ConvertAll(x => (double)x));
                lossSum += -Math.Log(predicted[m_TrainingLabels[j]]);
                batchCount++;
                dataCount++;
                //Debug.Log($"Predicted = {PrintList(predicted)}\nExpected = {PrintList(LabelToOutputValue(m_Labels[j]))}");
                //Debug.Log($"Predicted: {OutputToLabelValue(predicted)} Actual: {m_Labels[j]}");
                if (OutputToLabelValue(predicted) == m_TrainingLabels[j])
                {
                    m_NumberOfCorrectTraining++;
                }
                if (batchCount >= m_MiniBatchSize)
                {
                    ann.ApplyGradients(m_MiniBatchSize);
                    batchCount = 0;
                }

                if(dataCount >= m_DataPerPoint)
                {
                    m_LineRendererTraining.positionCount++;
                    m_LineRendererTraining.SetPosition(m_LineRendererTraining.positionCount - 1, new Vector3(m_LineRendererTraining.positionCount - 1, (float)(m_NumberOfCorrectTraining * 100) / m_DataPerPoint, 0));
                    m_LineRendererLoss.positionCount++;
                    m_LineRendererLoss.SetPosition(m_LineRendererLoss.positionCount - 1, new Vector3(m_LineRendererLoss.positionCount - 1, (float)lossSum / m_DataPerPoint, 0));
                    Debug.Log($"{j} {m_MiniBatchSize} {m_LineRendererTraining.positionCount - 1}");
                    m_NumberOfCorrectTraining = 0;
                    dataCount = 0;
                    lossSum = 0;

                    yield return null;
                }
            }
        }

        StartCoroutine(StartTesting());
    }
    IEnumerator StartTesting()
    {
        int miniBatchTest = m_MiniBatchSize / 10;
        int batchCount = 0;
        int dataCount = 0;
        int noOfIter = m_TestingImagesLoaded > m_TestingLabelsLoaded ? m_TestingLabelsLoaded : m_TestingImagesLoaded;
        for (int j = 0; j < noOfIter; j++)
        {
            double[] image;
            List<double> inputs = new();
            //for (int kels = 0; kels < m_Kernel.Length; kels++)
            {
                image = ImageProcessor.BlackWhiteImage(m_TrainingImageValues[j], m_BlackWhiteThreshold);
                /*image = ImageProcessor.KerneledImage(image, m_Kernel[kels]);
                image = ImageProcessor.MaxPool(image, 2);*/
                /*image = ImageProcessor.KerneledImage(image, m_Kernel[kels]);
                image = ImageProcessor.MaxPool(image, 2);*/

                for (int pxl = 0; pxl < image.Length; pxl++)
                {
                    inputs.Add(image[pxl]);
                }
            }
            List<double> predicted = ann.Test(inputs);
            batchCount++;
            dataCount++;
            //Debug.Log($"Predicted = {PrintList(predicted)}\nExpected = {PrintList(LabelToOutputValue(m_Labels[j]))}");
            //Debug.Log($"Predicted: {OutputToLabelValue(predicted)} Actual: {m_Labels[j]}");
            if (OutputToLabelValue(predicted) == m_TestingLabels[j])
            {
                m_NumberOfCorrectTesting++;
            }
            if (batchCount >= miniBatchTest)
            {
                batchCount = 0;
            }
            if (dataCount >= m_DataPerPoint)
            {
                m_LineRendererTesting.positionCount++;
                m_LineRendererTesting.SetPosition(m_LineRendererTesting.positionCount - 1, new Vector3(m_LineRendererTesting.positionCount - 1, (float)(m_NumberOfCorrectTesting * 100) / m_DataPerPoint, 0));
                Debug.Log($"{j} {m_MiniBatchSize} {m_LineRendererTesting.positionCount - 1}");
                m_NumberOfCorrectTesting = 0;
                dataCount = 0;

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

        if(idx == -1)
        {
            Debug.LogWarning("Index = -1");
            idx = 0;
            string str = "";
            for (int i = 0; i < value.Count; i++)
            {
                str += value[i].ToString() + " ";
            }

            Debug.Log("string : " + str);
        }

        return idx;
    }
}
