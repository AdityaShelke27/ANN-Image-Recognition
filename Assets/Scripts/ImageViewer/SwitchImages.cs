using System.Collections;
using UnityEngine;

public class SwitchImages : MonoBehaviour
{
    [SerializeField] Vector2 Position;
    [SerializeField] Vector2 Scale;
    [SerializeField] float Rotation;

    float[] m_Image;

    int m_ImagesLoaded;
    int m_ImageHeight;
    int m_ImageWidth;
    int m_LabelsLoaded;
    byte[][] m_Images;
    byte[] m_Labels;
    int m_CurrentImageIdx = 0;

    ComputeBuffer m_ImageBuffer;
    [SerializeField] RenderTexture m_ImageTexture;
    uint[] m_ComputeShaderThreadGroup = new uint[3];

    float[][] m_ImageValues;
    [SerializeField] float m_BlackWhiteThreshold;
    [SerializeField] float[] m_Kernel;
    [SerializeField] string m_TrainingImagePath;
    [SerializeField] string m_TrainingLabelPath;
    [SerializeField] int m_Resolution;
    [SerializeField] ComputeShader m_SetImageShader;
    [SerializeField] GameObject m_Canvas;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        m_ImageTexture = CreateTexture();
        m_Canvas.GetComponent<MeshRenderer>().material.mainTexture = m_ImageTexture;
        
        m_SetImageShader.GetKernelThreadGroupSizes(0, out m_ComputeShaderThreadGroup[0],
            out m_ComputeShaderThreadGroup[1], out m_ComputeShaderThreadGroup[2]);

        ImageSetup();
        //ImageTransformInitializer();
        ApplyImage(m_ImageValues[0]);
    }

    // Update is called once per frame
    void Update()
    {
        ApplyImage(m_Image);
    }
    void ImageSetup()
    {
        (m_ImagesLoaded, m_ImageHeight, m_ImageWidth, m_Images) =
            DataManager.LoadImages(m_TrainingImagePath, 0);

        (m_LabelsLoaded, m_Labels) =
            DataManager.LoadLabels(m_TrainingLabelPath, 0);

        m_ImageValues = new float[m_Images.Length][];
        for (int i = 0; i < m_Images.Length; i++)
        {
            m_ImageValues[i] = new float[m_Images[i].Length];
            for (int j = 0; j < m_Images[i].Length; j++)
            {
                m_ImageValues[i][j] = (float)m_Images[i][j] / 255;
            }
        }
    }
    void ApplyImage(float[] image)
    {
        image = ImageProcessor.BlackWhiteImage(image, m_BlackWhiteThreshold);
        image = ImageProcessor.TransformTexture(image, Rotation, Scale, Position);
        m_Image = image;
        /*image = ImageProcessor.KerneledImage(image, m_Kernel);
        image = ImageProcessor.MaxPool(image, 2);*/
        /*image = ImageProcessor.KerneledImage(image, m_Kernel);
        image = ImageProcessor.MaxPool(image, 2);*/

        m_Resolution = (int) Mathf.Sqrt(image.Length);
        m_ImageTexture = CreateTexture();
        m_Canvas.GetComponent<MeshRenderer>().material.mainTexture = m_ImageTexture;
        m_SetImageShader.SetTexture(0, "Result", m_ImageTexture);
        m_SetImageShader.SetInt("Resolution", m_Resolution);
        m_ImageBuffer = new ComputeBuffer(image.Length, sizeof(float));
        m_ImageBuffer.SetData(image);
        m_SetImageShader.SetBuffer(0, "Data", m_ImageBuffer);
        m_SetImageShader.Dispatch(0, (int)(m_Resolution / m_ComputeShaderThreadGroup[0]) + 1,
                (int)(m_Resolution / m_ComputeShaderThreadGroup[1]) + 1,
                (int)(m_Resolution / m_ComputeShaderThreadGroup[2]) + 1);
        m_ImageBuffer.Dispose();

       /* m_ImageTexture = TransformRenderTexture(m_ImageTexture, Rotation, Scale, Position);
        m_Canvas.GetComponent<MeshRenderer>().material.mainTexture = m_ImageTexture;*/
    }
    RenderTexture CreateTexture()
    {
        RenderTexture rt = new RenderTexture(m_Resolution, m_Resolution, 0);
        rt.filterMode = FilterMode.Point;
        rt.wrapMode = TextureWrapMode.Clamp;
        rt.enableRandomWrite = true;

        return rt;
    }

    public void NextImage()
    {
        m_CurrentImageIdx = (m_CurrentImageIdx + 1) % m_ImageValues.Length;

        ApplyImage(m_ImageValues[m_CurrentImageIdx]);
    }
    public void PreviousImage()
    {
        m_CurrentImageIdx--;
        if(m_CurrentImageIdx < 0)
        {
            m_CurrentImageIdx += m_ImageValues.Length;
        }

        ApplyImage(m_ImageValues[m_CurrentImageIdx]);
    }
}
