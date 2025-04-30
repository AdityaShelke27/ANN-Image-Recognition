using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Random = UnityEngine.Random;

public class DoodleTrainingViewer : MonoBehaviour
{
    public BoxCollider2D canvasCollider;
    [SerializeField] int m_Resolution;
    [SerializeField] int m_DataPerPoint;

    [SerializeField] RenderTexture m_ImageTexture;

    ANN ann;
    int m_TrainingImagesLoaded;
    int m_TrainingLabelsLoaded;
    double[][] m_TrainingImages;
    int[] m_TrainingLabels;
    int m_NumberOfCorrectTraining;
    [SerializeField] LineRenderer m_LineRendererTraining;

    int m_TestingImagesLoaded;
    int m_TestingLabelsLoaded;
    double[][] m_TestingImages;
    int[] m_TestingLabels;
    int m_NumberOfCorrectTesting;
    [SerializeField] LineRenderer m_LineRendererTesting;

    [SerializeField] LineRenderer m_LineRendererLoss;
    [SerializeField] int m_TotalClassifications;

    [Header("Training Parameters")]
    [SerializeField] string m_RootImageDirectory;
    [SerializeField] int m_Epochs;
    [SerializeField] int m_MiniBatchSize;
    [SerializeField, Range(0, 1)] float m_TrainTestSplit;
    [SerializeField] int m_DataShuffleIterations;
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

        SetupTrainingAndTestingImages();
        ShuffleDataset();

        StartCoroutine(StartTraining());
    }

    // Update is called once per frame
    void Update()
    {

    }

    IEnumerator StartTraining()
    {
        m_TrainingImagesLoaded = m_TrainingImages.Length;
        for (int i = 0; i < m_Epochs; i++)
        {
            int batchCount = 0;
            int dataCount = 0;
            double lossSum = 0;
            int batchLeftover = m_TrainingImagesLoaded % m_MiniBatchSize;
            int totalBatches = (m_TrainingImagesLoaded / m_MiniBatchSize) + (batchLeftover == 0 ? 0 : 1);
            for (int k = 0; k < totalBatches; k++)
            {
                int iters = 0;

                if(k == totalBatches - 1 && batchLeftover != 0) iters = batchLeftover;
                else iters = m_MiniBatchSize;

                System.Threading.Tasks.Parallel.For(0, iters, (i) =>
                {
                    int idx = (k * m_MiniBatchSize) + i;
                    double[] image;
                    List<double> inputs = new();

                    image = ImageProcessor.TransformTexture(m_TrainingImages[idx], Random.Range(m_RotationRandomizer.x, m_RotationRandomizer.y),
                            new Vector2(Random.Range(m_ScaleRandomizer.x, m_ScaleRandomizer.y), Random.Range(m_ScaleRandomizer.x, m_ScaleRandomizer.y)),
                            new Vector2(Random.Range(m_PositionRandomizer.x, m_PositionRandomizer.y), Random.Range(m_PositionRandomizer.x, m_PositionRandomizer.y)), 254);
                    for (int kels = 0; kels < m_Kernel.Length; kels++)
                    {
                        double[] kernelImage;
                        kernelImage = ImageProcessor.KerneledImage(image, m_Kernel[kels]);
                        kernelImage = ImageProcessor.MaxPool(kernelImage, 2);
                        kernelImage = ImageProcessor.KerneledImage(kernelImage, m_Kernel[kels]);
                        kernelImage = ImageProcessor.MaxPool(kernelImage, 2);
                        kernelImage = ImageProcessor.KerneledImage(kernelImage, m_Kernel[kels]);
                        kernelImage = ImageProcessor.MaxPool(kernelImage, 2);
                        kernelImage = ImageProcessor.KerneledImage(kernelImage, m_Kernel[kels]);
                        kernelImage = ImageProcessor.MaxPool(kernelImage, 2);

                        for (int pxl = 0; pxl < kernelImage.Length; pxl++)
                        {
                            inputs.Add(kernelImage[pxl]);
                        }
                    }
                    List<double> predicted = ann.Train(inputs, LabelToOutputValue(m_TrainingLabels[idx]));
                    lossSum += -Math.Log(predicted[(int)m_TrainingLabels[idx]]);
                    batchCount++;
                    dataCount++;
                    //Debug.Log($"Predicted = {PrintList(predicted)}\nExpected = {PrintList(LabelToOutputValue(m_Labels[j]))}");
                    //Debug.Log($"Predicted: {OutputToLabelValue(predicted)} Actual: {m_Labels[j]}");
                    if (OutputToLabelValue(predicted) == m_TrainingLabels[idx])
                    {
                        m_NumberOfCorrectTraining++;
                    }
                });

                ann.ApplyGradients(m_MiniBatchSize);
                if (dataCount >= m_DataPerPoint)
                {
                    m_LineRendererTraining.positionCount++;
                    m_LineRendererTraining.SetPosition(m_LineRendererTraining.positionCount - 1, new Vector3(m_LineRendererTraining.positionCount - 1, (float)(m_NumberOfCorrectTraining * 100) / m_DataPerPoint, 0));
                    m_LineRendererLoss.positionCount++;
                    m_LineRendererLoss.SetPosition(m_LineRendererLoss.positionCount - 1, new Vector3(m_LineRendererLoss.positionCount - 1, (float)lossSum / m_DataPerPoint, 0));
                    Debug.Log($"{k} {m_MiniBatchSize} {m_LineRendererTraining.positionCount - 1}");
                    m_NumberOfCorrectTraining = 0;
                    dataCount = 0;
                    lossSum = 0;

                    yield return null;
                }
            }
            /*for (int j = 0; j < m_TrainingImagesLoaded; j++)
            {
                double[] image;
                List<double> inputs = new();

                image = ImageProcessor.TransformTexture(m_TrainingImages[j], Random.Range(m_RotationRandomizer.x, m_RotationRandomizer.y),
                        new Vector2(Random.Range(m_ScaleRandomizer.x, m_ScaleRandomizer.y), Random.Range(m_ScaleRandomizer.x, m_ScaleRandomizer.y)),
                        new Vector2(Random.Range(m_PositionRandomizer.x, m_PositionRandomizer.y), Random.Range(m_PositionRandomizer.x, m_PositionRandomizer.y)), 254);
                for (int kels = 0; kels < m_Kernel.Length; kels++)
                {
                    double[] kernelImage;
                    kernelImage = ImageProcessor.KerneledImage(image, m_Kernel[kels]);
                    kernelImage = ImageProcessor.MaxPool(kernelImage, 2);
                    kernelImage = ImageProcessor.KerneledImage(kernelImage, m_Kernel[kels]);
                    kernelImage = ImageProcessor.MaxPool(kernelImage, 2);
                    kernelImage = ImageProcessor.KerneledImage(kernelImage, m_Kernel[kels]);
                    kernelImage = ImageProcessor.MaxPool(kernelImage, 2);
                    kernelImage = ImageProcessor.KerneledImage(kernelImage, m_Kernel[kels]);
                    kernelImage = ImageProcessor.MaxPool(kernelImage, 2);

                    for (int pxl = 0; pxl < kernelImage.Length; pxl++)
                    {
                        inputs.Add(kernelImage[pxl]);
                    }
                }
                List<double> predicted = ann.Train(inputs, LabelToOutputValue(m_TrainingLabels[j]));
                lossSum += -Math.Log(predicted[(int)m_TrainingLabels[j]]);
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

                if (dataCount >= m_DataPerPoint)
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
            }*/
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
            List<double> inputs = new();
            for (int kels = 0; kels < m_Kernel.Length; kels++)
            {
                double[] kernelImage;
                kernelImage = ImageProcessor.KerneledImage(m_TrainingImages[j], m_Kernel[kels]);
                kernelImage = ImageProcessor.MaxPool(kernelImage, 2);
                kernelImage = ImageProcessor.KerneledImage(kernelImage, m_Kernel[kels]);
                kernelImage = ImageProcessor.MaxPool(kernelImage, 2);
                kernelImage = ImageProcessor.KerneledImage(kernelImage, m_Kernel[kels]);
                kernelImage = ImageProcessor.MaxPool(kernelImage, 2);
                kernelImage = ImageProcessor.KerneledImage(kernelImage, m_Kernel[kels]);
                kernelImage = ImageProcessor.MaxPool(kernelImage, 2);

                for (int pxl = 0; pxl < kernelImage.Length; pxl++)
                {
                    inputs.Add(kernelImage[pxl]);
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
    void SetupTrainingAndTestingImages()
    {
        string[] imagepath = Directory.GetFiles(Application.dataPath + m_RootImageDirectory);

        bool[][] m_LoadedImages = new bool[3000 * imagepath.Length / 4][];
        int count = 0;
        for (int j = 0; j < imagepath.Length; j++)
        {
            if (Path.GetExtension(imagepath[j]) == ".meta") continue;

            byte[] arr = File.ReadAllBytes(imagepath[j]);
            using BinaryReader imgReader = new BinaryReader(new MemoryStream(arr));
            for (int k = 0; k < 1500; k++)
            {
                bool[] image = new bool[254 * 254];
                for (int i = 0; i < 254 * 254; i++)
                {
                    image[i] = imgReader.ReadBoolean();
                }
                m_LoadedImages[count] = image;
                count++;
            }
        }
        Debug.Log("Images Loaded");
        int splitPoint = Mathf.FloorToInt(3000 * m_TrainTestSplit);
        m_TrainingImages = new double[splitPoint * m_TotalClassifications][];
        m_TrainingLabels = new int[splitPoint * m_TotalClassifications];
        m_TestingImages = new double[(3000 - splitPoint) * m_TotalClassifications][];
        m_TestingLabels = new int[(3000 - splitPoint) * m_TotalClassifications];
        count = 0;
        int countTest = 0;
        for (int i = 0; i < m_TotalClassifications; i++)
        {
            for (int j = 0; j < 3000; j++)
            {
                int idx = (i * 3000) + j;
                double[] convertToDouble = new double[m_LoadedImages[idx].Length];
                for (int it = 0; it < convertToDouble.Length; it++)
                {
                    convertToDouble[it] = m_LoadedImages[idx][it] ? 1 : 0;
                }
                if (j < splitPoint)
                {
                    m_TrainingImages[count] = convertToDouble;
                    m_TrainingLabels[count] = i;
                    count++;
                }
                else
                {
                    m_TestingImages[countTest] = convertToDouble;
                    m_TestingLabels[countTest] = i;
                    countTest++;
                }
            }
        }
        Debug.Log("Images Splitted");
    }
    void ShuffleDataset()
    {
        for (int i = 0; i < m_DataShuffleIterations; i++)
        {
            int idx1 = Random.Range(0, m_TrainingImages.Length);
            int idx2 = Random.Range(0, m_TrainingImages.Length);

            double[] temp = m_TrainingImages[idx1];
            m_TrainingImages[idx1] = m_TrainingImages[idx2];
            m_TrainingImages[idx2] = temp;

            int tempL = m_TrainingLabels[idx1];
            m_TrainingLabels[idx1] = m_TrainingLabels[idx2];
            m_TrainingLabels[idx2] = tempL;
        }
    }
    List<double> LabelToOutputValue(int value)
    {
        List<double> output = new();
        for (int i = 0; i < m_TotalClassifications; i++)
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

        if (idx == -1)
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
