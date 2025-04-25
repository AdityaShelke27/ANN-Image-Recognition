using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class PNGViewer : MonoBehaviour
{
    ComputeBuffer m_ImageBuffer;
    [SerializeField] string m_FilePath;
    [SerializeField] int m_Resolution;
    [SerializeField] int m_KernelNum;
    [SerializeField] string m_FileName;
    RenderTexture m_ImageTexture;
    Texture2D m_Image;
    float[][] m_Images;
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

        SaveImageInBinaryFromInt(m_FilePath, m_FileName);
        //TempFunc();
        //ApplyImage(m_Images[m_CurrentImageIdx]);
    }

    // Update is called once per frame
    void Update()
    {

    }
    void ApplyImage(float[] image)
    {
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
        byte[] arr = File.ReadAllBytes(Application.dataPath + $"/{m_FilePath}");
        using BinaryReader imgReader = new BinaryReader(new MemoryStream(arr));
        m_Images = new float[3000][];
        for (int count = 0; count < m_Images.Length; count++)
        {
            float[] image = new float[254 * 254];
            for (int i = 0; i < 254 * 254; i++)
            {
                image[i] = imgReader.ReadInt32();
            }
            m_Images[count] = image;
        }
    }
    public void NextImage()
    {
        m_CurrentImageIdx = (m_CurrentImageIdx + 1) % m_Images.Length;
        ApplyImage(m_Images[m_CurrentImageIdx]);
    }
    public void PreviousImage()
    {
        m_CurrentImageIdx--;
        if (m_CurrentImageIdx < 0)
        {
            m_CurrentImageIdx += m_Images.Length;
        }

        ApplyImage(m_Images[m_CurrentImageIdx]);
    }

    void SaveImageInBinary(string[] allFileNames, string fileName, Texture2D imageTex)
    {
        List<byte> bytes = new List<byte>();
        for (int j = 0; j < allFileNames.Length; j++)
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
        File.WriteAllBytes(Application.dataPath + $"/{fileName}", arr);
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
