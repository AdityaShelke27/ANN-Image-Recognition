using System.Collections;
using UnityEngine;
using UnityEngine.UI;

public class Painter : MonoBehaviour
{
    public BoxCollider2D canvasCollider;
    [SerializeField] int m_Resolution;
    [SerializeField] float m_SmoothStep;
    [SerializeField] float m_Strength;
    [SerializeField] ComputeShader m_ComputeShader;
    uint[] m_ComputeShaderThreadGroup = new uint[3];
    public RenderTexture m_DrawTexture;
    Vector2 m_LastPoint = new Vector2();
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        m_DrawTexture = ImageProcessor.CreateTexture(m_Resolution);//CreateTexture();
        canvasCollider.GetComponent<MeshRenderer>().material.mainTexture = m_DrawTexture;
        m_ComputeShader.GetKernelThreadGroupSizes(0, out m_ComputeShaderThreadGroup[0], 
            out m_ComputeShaderThreadGroup[1], out m_ComputeShaderThreadGroup[2]);

        m_ComputeShader.SetTexture(0, "Result", m_DrawTexture);
        m_ComputeShader.SetFloat("SmoothStep", m_SmoothStep);
        m_ComputeShader.SetFloat("Strength", m_Strength);

    }

    // Update is called once per frame
    void Update()
    {
        
        if (Input.GetMouseButtonDown(0))
        {
            Vector3 pos = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            Vector2 pixelPos = new Vector2(Mathf.InverseLerp(canvasCollider.bounds.min.x, canvasCollider.bounds.max.x, pos.x),
                Mathf.InverseLerp(canvasCollider.bounds.min.y, canvasCollider.bounds.max.y, pos.y)) * m_Resolution;

            m_LastPoint = pixelPos;
        }
        if(Input.GetMouseButton(0))
        {
            Vector3 pos = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            Vector2 pixelPos = new Vector2(Mathf.InverseLerp(canvasCollider.bounds.min.x, canvasCollider.bounds.max.x, pos.x), 
                Mathf.InverseLerp(canvasCollider.bounds.min.y, canvasCollider.bounds.max.y, pos.y)) * m_Resolution;

            m_ComputeShader.SetFloats("CurrentPosition", (int)pixelPos.x, (int)pixelPos.y);
            m_ComputeShader.SetFloats("LastPosition", (int)m_LastPoint.x, (int)m_LastPoint.y);
            m_ComputeShader.SetFloats("Resolution", m_Resolution);
            
            float spacing = m_SmoothStep * 0.5f;
            float dist = Vector2.Distance(m_LastPoint, pixelPos);
            int steps = Mathf.CeilToInt(dist / spacing);
            for(int i = 0; i <= steps; i++)
            {
                Vector2 point = Vector2.Lerp(m_LastPoint, pixelPos, i / (float)steps);
                int minX = Mathf.Max(0, Mathf.FloorToInt(point.x - m_SmoothStep));
                int maxX = Mathf.Min(m_Resolution, Mathf.CeilToInt(point.x + m_SmoothStep));
                int minY = Mathf.Max(0, Mathf.FloorToInt(point.y - m_SmoothStep));
                int maxY = Mathf.Min(m_Resolution, Mathf.CeilToInt(point.y + m_SmoothStep));

                int regionWidth = Mathf.Max(1, maxX - minX);
                int regionHeight = Mathf.Max(1, maxY - minY);

                int dispatchX = Mathf.CeilToInt(regionWidth / 8f);
                int dispatchY = Mathf.CeilToInt(regionHeight / 8f);
                m_ComputeShader.SetInts("offset", minX, minY);
                m_ComputeShader.SetVector("circleCenter", point);
                /*m_ComputeShader.Dispatch(0, (int)(m_Resolution / m_ComputeShaderThreadGroup[0]) + 1, 
                (int)(m_Resolution / m_ComputeShaderThreadGroup[1]) + 1, 
                (int)(m_Resolution / m_ComputeShaderThreadGroup[2]) + 1);*/
                m_ComputeShader.Dispatch(0, dispatchX, dispatchY, 1);
            }

            m_LastPoint = pixelPos;
        }

        if (Input.GetKeyDown(KeyCode.C))
        {
            ClearTexture();
        }
    }
    void ClearTexture()
    {
        RenderTexture texture = ImageProcessor.CreateTexture(m_Resolution);

        Graphics.Blit(texture, m_DrawTexture);
        texture.Release();
    }
}
