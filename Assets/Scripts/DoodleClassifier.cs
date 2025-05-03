using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;
using Random = UnityEngine.Random;

public class DoodleClassifier : MonoBehaviour
{
    ANN ann;
    int m_ImagesLoaded;
    int m_ImageHeight;
    int m_ImageWidth;
    int m_LabelsLoaded;
    byte[][] m_Images;
    byte[] m_Labels;

    Dictionary<int, string> m_IndexToLabel = new () { 
        { 0, "Apple"},
        { 1, "Bicycle"},
        { 2, "Butterfly"},
        { 3, "Carrot"},
        { 4, "Clock"},
        { 5, "Cup"},
        { 6, "Duck"},
        { 7, "Hammer"},
        { 8, "Hourglass"},
        { 9, "Lighthouse"},
        { 10, "Popsicle"},
        { 11, "Saw"},
        { 12, "Smiley Face"},
        { 13, "Sun"},
        { 14, "Sword"},
        { 15, "Truck"},
    };

    double[][] m_ImageValues;
    Texture2D texture;
    double[][] m_Kernel = new double[][] {
        new double[] { 1, 2, 1, 0, 0, 0, -1, -2, -1 },
        new double[] { -1, -2, -1, 0, 0, 0, 1, 2, 1 },
        new double[] { 1, 0, -1, 2, 0, -2, 1, 0, -1 },
        new double[] { -1, 0, 1, -2, 0, 2, -1, 0, 1 },
    };
    [SerializeField] Painter m_Painter;
    [SerializeField] TMP_Text m_Text;

    [Header("Training Parameters")]
    [SerializeField] string m_TrainingImagePath;
    [SerializeField] int m_Epochs;
    [SerializeField] int m_MiniBatchSize;
    [SerializeField] Vector2 m_RotationRandomizer;
    [SerializeField] Vector2 m_PositionRandomizer;
    [SerializeField] Vector2 m_ScaleRandomizer;
    [SerializeField] int m_DataShuffleIterations;

    [Header("ANN Parameters")]
    [SerializeField] int m_NoOfInputs;
    [SerializeField] int m_NoOfOutputs;
    [SerializeField] int m_NoOfHiddenLayers;
    [SerializeField] int m_NoOfNeuronsPerHiddenLayers;
    [SerializeField] int m_TotalClassifications;
    [SerializeField] float m_LearningRate;
    [SerializeField] double m_RegularizationFactor;
    [SerializeField] Activation m_HiddenActivation;
    [SerializeField] Activation m_OutputActivation;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        texture = new Texture2D(254, 254, TextureFormat.RGBA32, false);
        //RenderTexture.active = m_Painter.m_DrawTexture;
        ann = new ANN(m_NoOfInputs, m_NoOfOutputs, m_NoOfHiddenLayers, m_NoOfNeuronsPerHiddenLayers,
            m_LearningRate, m_HiddenActivation, m_OutputActivation, m_RegularizationFactor, m_MiniBatchSize);
        SetupTrainingImages();
        ShuffleDataset();
        StartTraining();

        StartCoroutine(PredictCanvas());
        Debug.Log("Done");
    }

    // Update is called once per frame
    void Update()
    {

    }
    void SetupTrainingImages()
    {
        string[] imagepath = Directory.GetFiles(Application.dataPath + m_TrainingImagePath);

        m_Images = new byte[3000 * m_TotalClassifications][];
        m_Labels = new byte[3000 * m_TotalClassifications];
        int count = 0;
        byte classifier = 0;
        for (int j = 0; j < imagepath.Length; j++)
        {
            if (Path.GetExtension(imagepath[j]) == ".meta") continue;

            byte[] arr = File.ReadAllBytes(imagepath[j]);
            using BinaryReader imgReader = new BinaryReader(new MemoryStream(arr));
            for (int k = 0; k < 1500; k++)
            {
                byte[] image = new byte[254 * 254];
                for (int i = 0; i < 254 * 254; i++)
                {
                    image[i] = (byte) (imgReader.ReadBoolean() ? 1 : 0);
                }
                m_Images[count] = image;
                m_Labels[count] = (byte) (classifier / 2);
                count++;
            }
            classifier++;
        }
        Debug.Log("Images Loaded");
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

        double[] image;
        List<double> inputs = new();
        image = pixels.ToArray();
        for (int kels = 0; kels < m_Kernel.Length; kels++)
        {
            double[] kerneledImage;
            kerneledImage = ImageProcessor.KerneledImage(image, m_Kernel[kels]);
            kerneledImage = ImageProcessor.MaxPool(kerneledImage, 2);
            kerneledImage = ImageProcessor.KerneledImage(kerneledImage, m_Kernel[kels]);
            kerneledImage = ImageProcessor.MaxPool(kerneledImage, 2);
            kerneledImage = ImageProcessor.KerneledImage(kerneledImage, m_Kernel[kels]);
            kerneledImage = ImageProcessor.MaxPool(kerneledImage, 2);
            kerneledImage = ImageProcessor.KerneledImage(kerneledImage, m_Kernel[kels]);
            kerneledImage = ImageProcessor.MaxPool(kerneledImage, 2);

            for (int pxl = 0; pxl < kerneledImage.Length; pxl++)
            {
                inputs.Add(kerneledImage[pxl]);
            }
        }
        Debug.Log(inputs.Count);
        List<double> predicted = ann.Test(inputs);

        m_Text.text = m_IndexToLabel[OutputToLabelValue(predicted)];
        //Debug.Log(OutputToLabelValue(predicted));
        StartCoroutine(PredictCanvas());
    }
    void ShuffleDataset()
    {
        for (int i = 0; i < m_DataShuffleIterations; i++)
        {
            int idx1 = Random.Range(0, m_Images.Length);
            int idx2 = Random.Range(0, m_Images.Length);

            byte[] temp = m_Images[idx1];
            m_Images[idx1] = m_Images[idx2];
            m_Images[idx2] = temp;

            byte tempL = m_Labels[idx1];
            m_Labels[idx1] = m_Labels[idx2];
            m_Labels[idx2] = tempL;
        }
    }
    public static double[] ToDoubleArray(byte[] input)
    {
        double[] result = new double[input.Length];
        for (int i = 0; i < input.Length; i++)
            result[i] = input[i];
        return result;
    }
    void StartTraining()
    {
        int m_TrainingImagesLoaded = m_Images.Length;
        for (int i = 0; i < m_Epochs; i++)
        {
            int batchCount = 0;
            for (int j = 0; j < m_TrainingImagesLoaded; j++)
            {

                double[] image;
                List<double> inputs = new();

                image = ImageProcessor.TransformTexture(ToDoubleArray(m_Images[j]), Random.Range(m_RotationRandomizer.x, m_RotationRandomizer.y),
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
                List<double> predicted = ann.Train(inputs, LabelToOutputValue(m_Labels[j]).ConvertAll(x => (double)x));
                batchCount++;
                if (batchCount >= m_MiniBatchSize)
                {
                    ann.ApplyGradients(m_MiniBatchSize);
                    batchCount = 0;
                }
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

        if(idx == -1)
        {
            Debug.Log("Idx is -1");
            return 0;
        }

        return idx;
    }
}
