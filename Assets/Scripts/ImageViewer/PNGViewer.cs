using System;
using System.Collections.Generic;
using System.IO;
using Unity.VisualScripting;
using UnityEngine;

public class PNGViewer : MonoBehaviour
{
    ComputeBuffer m_ImageBuffer;
    [SerializeField] string m_RootImageDirectory;
    [SerializeField] int m_Resolution;
    [SerializeField] string m_FileName;
    [SerializeField, Range(0, 1)] float m_TrainTestSplit;
    [SerializeField] int m_TotalClassifications;
    float[][] m_TrainingImages;
    int[] m_TrainingLabels;
    float[][] m_TestingImages;
    int[] m_TestingLabels;
    RenderTexture m_ImageTexture;
    Texture2D m_Image;
    uint[] m_ComputeShaderThreadGroup = new uint[3];
    int m_CurrentImageIdx = 0;
    float[][] m_Kernel = new float[][] {
        new float[] { 1, 2, 1, 0, 0, 0, -1, -2, -1 },
        new float[] { -1, -2, -1, 0, 0, 0, 1, 2, 1 },
        new float[] { 1, 0, -1, 2, 0, -2, 1, 0, -1 },
        new float[] { -1, 0, 1, -2, 0, 2, -1, 0, 1 },
    };

    [SerializeField] ComputeShader m_SetImageShader;
    [SerializeField] GameObject m_Canvas;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        m_ImageTexture = ImageProcessor.CreateTexture(m_Resolution);
        m_Canvas.GetComponent<MeshRenderer>().material.mainTexture = m_ImageTexture;

        m_SetImageShader.GetKernelThreadGroupSizes(0, out m_ComputeShaderThreadGroup[0],
            out m_ComputeShaderThreadGroup[1], out m_ComputeShaderThreadGroup[2]);

        m_Image = new Texture2D(m_Resolution, m_Resolution);

        //SaveImageInBinary(m_RootImageDirectory, m_FileName, new Texture2D(254, 254));
        //TempFunc();
        SetupTrainingAndTestingImages();
        ApplyImage(m_TrainingImages[m_CurrentImageIdx]);
    }

    // Update is called once per frame
    void Update()
    {

    }
    void ApplyImage(float[] image)
    {
        image = ImageProcessor.KerneledImage(image, m_Kernel[0]);
        image = ImageProcessor.MaxPool(image, 2);
        image = ImageProcessor.KerneledImage(image, m_Kernel[0]);
        image = ImageProcessor.MaxPool(image, 2);
        image = ImageProcessor.KerneledImage(image, m_Kernel[0]);
        image = ImageProcessor.MaxPool(image, 2);
        image = ImageProcessor.KerneledImage(image, m_Kernel[0]);
        image = ImageProcessor.MaxPool(image, 2);

        m_Resolution = (int)Mathf.Sqrt(image.Length);
        m_ImageTexture = ImageProcessor.CreateTexture(m_Resolution);
        m_SetImageShader.SetTexture(0, "Result", m_ImageTexture);
        m_SetImageShader.SetInt("Resolution", m_Resolution);
        m_ImageBuffer = new ComputeBuffer(image.Length, sizeof(float));
        m_ImageBuffer.SetData(image);
        m_SetImageShader.SetBuffer(0, "Data", m_ImageBuffer);
        m_SetImageShader.Dispatch(0, (int)(m_Resolution / m_ComputeShaderThreadGroup[0]) + 1,
                (int)(m_Resolution / m_ComputeShaderThreadGroup[1]) + 1,
                (int)(m_Resolution / m_ComputeShaderThreadGroup[2]));
        m_ImageBuffer.Dispose();
        m_Canvas.GetComponent<MeshRenderer>().material.mainTexture = m_ImageTexture;
    }
    void TempFunc()
    {
        string[] imagepath = Directory.GetFiles(Application.dataPath + m_RootImageDirectory);

        float[][] m_LoadedImages = new float[3000 * imagepath.Length / 2][];
        int count = 0;
        bool rep = false;
        for (int j = 0; j < imagepath.Length; j++)
        {
            if (Path.GetExtension(imagepath[j]) == ".meta") continue;

            byte[] arr = File.ReadAllBytes(imagepath[j]);
            using BinaryReader imgReader = new BinaryReader(new MemoryStream(arr));
            for (int k = 0; k < 3000; k++)
            {
                float[] image = new float[254 * 254];
                for (int i = 0; i < 254 * 254; i++)
                {
                    int num = imgReader.ReadInt32();
                    if(!rep)
                    {
                        Debug.Log(num);
                    }
                    if (num == 1)
                    {
                        image[i] = 1;
                    }
                    else
                    {
                        image[i] = 0;
                    }
                }
                if(!rep)
                {
                    rep = true;
                }
                m_LoadedImages[count] = image;
                count++;
            }
        }

        m_TrainingImages = m_LoadedImages;
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
        m_TrainingImages = new float[splitPoint * m_TotalClassifications][];
        m_TrainingLabels = new int[splitPoint * m_TotalClassifications];
        m_TestingImages = new float[(3000 - splitPoint) * m_TotalClassifications][];
        m_TestingLabels = new int[(3000 - splitPoint) * m_TotalClassifications];
        count = 0;
        int countTest = 0;
        for (int i = 0; i < m_TotalClassifications; i++)
        {
            for (int j = 0; j < 3000; j++)
            {
                int idx = (i * 3000) + j;
                
                float[] convertToDouble = new float[m_LoadedImages[idx].Length];
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
    public void NextImage()
    {
        m_CurrentImageIdx = (m_CurrentImageIdx + 1) % m_TrainingImages.Length;
        ApplyImage(m_TrainingImages[m_CurrentImageIdx]);
    }
    public void PreviousImage()
    {
        m_CurrentImageIdx--;
        if (m_CurrentImageIdx < 0)
        {
            m_CurrentImageIdx += m_TrainingImages.Length;
        }

        ApplyImage(m_TrainingImages[m_CurrentImageIdx]);
    }

    void SaveImageInBinary(string filePath, string fileName, Texture2D imageTex)
    {
        List<byte> bytes = new List<byte>();
        string[] allFileNames = Directory.GetFiles(Application.dataPath + filePath);
        for (int j = 0; j < allFileNames.Length / 2; j++)
        {
            string path = allFileNames[j];

            if (File.Exists(path))
            {
                if (Path.GetExtension(path) != ".png")
                {
                    continue;
                }
                byte[] data = File.ReadAllBytes(path);
                ImageConversion.LoadImage(imageTex, data);

                Color[] cols = imageTex.GetPixels();
                bool[] image = new bool[cols.Length];
                for (int i = 0; i < cols.Length; i++)
                {
                    if (cols[i].r < 1)
                    {
                        image[i] = true;
                    }
                    else
                    {
                        image[i] = false;
                    }
                    bytes.AddRange(BitConverter.GetBytes(image[i]));
                    //Debug.Log(cols[i]);
                }
            }
        }
        byte[] arr = bytes.ToArray();
        File.WriteAllBytes(Application.dataPath + $"/{fileName}1", arr);
        bytes.Clear();
        for (int j = allFileNames.Length / 2; j < allFileNames.Length; j++)
        {
            string path = allFileNames[j];

            if (File.Exists(path))
            {
                if (Path.GetExtension(path) != ".png")
                {
                    continue;
                }
                byte[] data = File.ReadAllBytes(path);
                ImageConversion.LoadImage(imageTex, data);

                Color[] cols = imageTex.GetPixels();
                bool[] image = new bool[cols.Length];
                for (int i = 0; i < cols.Length; i++)
                {
                    if (cols[i].r < 1)
                    {
                        image[i] = true;
                    }
                    else
                    {
                        image[i] = false;
                    }
                    bytes.AddRange(BitConverter.GetBytes(image[i]));
                    //Debug.Log(cols[i]);
                }
            }
        }

        arr = bytes.ToArray();
        File.WriteAllBytes(Application.dataPath + $"/{fileName}2", arr);
        Debug.Log("Done");
    }
    void SaveImageInBinaryFromInt(string allFileName, string fileName)
    {
        List<byte> bytes = new List<byte>();
        string path = Application.dataPath + allFileName;

        if (File.Exists(path))
        {
            byte[] data = File.ReadAllBytes(path);

            using BinaryReader imgReader = new BinaryReader(new MemoryStream(data));
            for (int iter = 0; iter < 2; iter++)
            {
                for (int k = 0; k < 1500; k++)
                {
                    bool[] image = new bool[254 * 254];

                    for (int i = 0; i < image.Length; i++)
                    {
                        if (imgReader.ReadInt32() == 1)
                        {
                            image[i] = true;
                        }
                        else
                        {
                            image[i] = false;
                        }
                        bytes.AddRange(BitConverter.GetBytes(image[i]));
                        //Debug.Log(cols[i]);
                    }
                }

                byte[] arr = bytes.ToArray();
                File.WriteAllBytes(Application.dataPath + $"/{fileName}{iter + 1}.bin", arr);
                bytes.Clear();
            }
            Debug.Log("Done");
        }

    }
}
