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

    Texture2D texture;
    GameObject camGO;
    GameObject quad;
    Material mat;
    [SerializeField] RenderTexture rt;
    Camera cam;
    MeshRenderer mr;
    [SerializeField] ComputeShader m_SetImageShader;

    Vector3 OriginalPosition;
    Vector3 OriginalRotation;
    Vector3 OriginalScale;

    ComputeBuffer m_ImageBuffer;
    [SerializeField] RenderTexture m_ImageTexture;
    uint[] m_ComputeShaderThreadGroup = new uint[3];

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
        texture = new Texture2D(28, 28, TextureFormat.RGBA32, false);

        SetupTrainingImages();
        SetupTestingImages();
        ImageTransformInitializer();
        m_SetImageShader.GetKernelThreadGroupSizes(0, out m_ComputeShaderThreadGroup[0],
            out m_ComputeShaderThreadGroup[1], out m_ComputeShaderThreadGroup[2]);
        StartCoroutine(StartTraining());
    }

    // Update is called once per frame
    void Update()
    {

    }

    void ImageTransformInitializer()
    {
        // 1. Create a temporary Camera
        camGO = new GameObject("TempCam");
        cam = camGO.AddComponent<Camera>();
        cam.orthographic = true;
        cam.orthographicSize = 5f;
        cam.clearFlags = CameraClearFlags.Color;
        cam.backgroundColor = Color.black;

        quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        quad.transform.localScale = new Vector3(9, 9, 1);
        mat = new Material(Shader.Find("Unlit/Texture"));
        mr = quad.GetComponent<MeshRenderer>();

        rt = new RenderTexture(28, 28, 0, RenderTextureFormat.ARGB32);
        rt.filterMode = FilterMode.Point;
        rt.wrapMode = TextureWrapMode.Clamp;

        cam.transform.position = new Vector3(0f, 0f, -10f);

        mr.material = mat;

        OriginalPosition = quad.transform.position;
        OriginalRotation = quad.transform.rotation.eulerAngles;
        OriginalScale = quad.transform.localScale;
    }
    RenderTexture CreateTexture()
    {
        RenderTexture rt = new RenderTexture(m_Resolution, m_Resolution, 0);
        rt.filterMode = FilterMode.Point;
        rt.wrapMode = TextureWrapMode.Clamp;
        rt.enableRandomWrite = true;

        return rt;
    }
    double[] ApplyImage(double[] image)
    {
        /*image = ImageProcessor.KerneledImage(image, m_Kernel);
        image = ImageProcessor.MaxPool(image, 2);*/
        float[] imageNew = new float[image.Length];
        for (int i = 0; i < image.Length; i++)
        {
            imageNew[i] = (float)image[i];
        }
        m_Resolution = (int)Mathf.Sqrt(image.Length);
        m_ImageTexture = CreateTexture();
        m_SetImageShader.SetTexture(0, "Result", m_ImageTexture);
        m_SetImageShader.SetInt("Resolution", m_Resolution);
        m_ImageBuffer = new ComputeBuffer(image.Length, sizeof(float));
        m_ImageBuffer.SetData(imageNew);
        m_SetImageShader.SetBuffer(0, "Data", m_ImageBuffer);
        m_SetImageShader.Dispatch(0, (int)(m_Resolution / m_ComputeShaderThreadGroup[0]) + 1,
                (int)(m_Resolution / m_ComputeShaderThreadGroup[1]) + 1,
                (int)(m_Resolution / m_ComputeShaderThreadGroup[2]) + 1);
        m_ImageBuffer.Dispose();

        m_ImageTexture = TransformRenderTexture(m_ImageTexture, Random.Range(-5f, 5f), OriginalScale * Random.Range(0.9f, 1.1f), OriginalPosition + new Vector3(Random.Range(-1f, 1f), Random.Range(-1f, 1f)));

        RenderTexture.active = m_ImageTexture;
        texture.ReadPixels(new Rect(0, 0, m_ImageTexture.width, m_ImageTexture.height), 0, 0);
        texture.Apply();
        RenderTexture.active = null;
        List<double> pixels = new();
        Color[] colArr = texture.GetPixels(0, 0, texture.width, texture.height);
        for (int i = 0; i < colArr.Length; i++)
        {
            pixels.Add((double)colArr[i].r);
        }

        return pixels.ToArray();
    }
    public RenderTexture TransformRenderTexture(RenderTexture source, float rotationDegrees, Vector2 scale, Vector2 position, int outputSize = 28)
    {
        mat.mainTexture = m_ImageTexture;
        quad.transform.position = new Vector3(position.x, position.y, 0f);
        quad.transform.localScale = new Vector3(scale.x, scale.y, 1f);
        quad.transform.rotation = Quaternion.Euler(0f, 0f, rotationDegrees);
        cam.targetTexture = rt;
        cam.Render();

        return rt;
    }
    IEnumerator StartTraining()
    {
        for (int i = 0; i < m_Epochs; i++)
        {
            int batchCount = 0;
            int dataCount = 0;
            int noOfIter = m_TrainingImagesLoaded > m_TrainingLabelsLoaded ? m_TrainingLabelsLoaded : m_TrainingImagesLoaded;
            for (int j = 0; j < noOfIter; j++)
            {
                double[] image;
                List<double> inputs = new();
                //for (int kels = 0; kels < m_Kernel.Length; kels++)
                {
                    image = ImageProcessor.BlackWhiteImage(m_TrainingImageValues[j], m_BlackWhiteThreshold);
                    image = ApplyImage(image);
                    yield return null;
                    /*image = ImageProcessor.KerneledImage(image, m_Kernel[kels]);
                    image = ImageProcessor.MaxPool(image, 2);*/
                    /*image = ImageProcessor.KerneledImage(image, m_Kernel[kels]);
                    image = ImageProcessor.MaxPool(image, 2);*/

                    for (int pxl = 0; pxl < image.Length; pxl++)
                    {
                        inputs.Add(image[pxl]);
                    }
                }

                List<double> predicted = ann.Train(inputs, LabelToOutputValue(m_TrainingLabels[j]).ConvertAll(x => (double)x));
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
                    Debug.Log($"{j} {m_MiniBatchSize} {m_LineRendererTraining.positionCount - 1}");
                    m_NumberOfCorrectTraining = 0;
                    dataCount = 0;

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

        return idx;
    }
}
