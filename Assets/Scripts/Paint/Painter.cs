using UnityEngine;
using UnityEngine.UI;

public class Painter : MonoBehaviour
{
    [SerializeField] BoxCollider2D m_CanvasCollider;
    [SerializeField] int m_Resolution;
    [SerializeField] float m_SmoothStep;
    [SerializeField] ComputeShader m_ComputeShader;
    uint[] m_ComputeShaderThreadGroup = new uint[3];
    RenderTexture m_DrawTexture;
    Vector2 m_LastPoint = new Vector2();
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        m_DrawTexture = CreateTexture();
        m_CanvasCollider.GetComponent<MeshRenderer>().material.mainTexture = m_DrawTexture;
        m_ComputeShader.GetKernelThreadGroupSizes(0, out m_ComputeShaderThreadGroup[0], 
            out m_ComputeShaderThreadGroup[1], out m_ComputeShaderThreadGroup[2]);

        m_ComputeShader.SetTexture(0, "Result", m_DrawTexture);
        m_ComputeShader.SetFloat("SmoothStep", m_SmoothStep);
    }

    // Update is called once per frame
    void Update()
    {
        if(Input.GetMouseButtonDown(0))
        {
            Vector3 pos = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            Vector2 pixelPos = new Vector2(Mathf.InverseLerp(m_CanvasCollider.bounds.min.x, m_CanvasCollider.bounds.max.x, pos.x),
                Mathf.InverseLerp(m_CanvasCollider.bounds.min.y, m_CanvasCollider.bounds.max.y, pos.y)) * m_Resolution;

            m_LastPoint = pixelPos;
        }
        if(Input.GetMouseButton(0))
        {
            Vector3 pos = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            Vector2 pixelPos = new Vector2(Mathf.InverseLerp(m_CanvasCollider.bounds.min.x, m_CanvasCollider.bounds.max.x, pos.x), 
                Mathf.InverseLerp(m_CanvasCollider.bounds.min.y, m_CanvasCollider.bounds.max.y, pos.y)) * m_Resolution;

            m_ComputeShader.SetFloats("CurrentPosition", (int)pixelPos.x, (int)pixelPos.y);
            m_ComputeShader.SetFloats("LastPosition", (int)m_LastPoint.x, (int)m_LastPoint.y);
            m_ComputeShader.Dispatch(0, (int)(m_Resolution / m_ComputeShaderThreadGroup[0]), 
                (int)(m_Resolution / m_ComputeShaderThreadGroup[1]), 
                (int)(m_Resolution / m_ComputeShaderThreadGroup[2]));


            m_LastPoint = pixelPos;
        }

        if (Input.GetKeyDown(KeyCode.C))
        {
            ClearTexture();
        }
    }

    RenderTexture CreateTexture()
    {
        RenderTexture rt = new RenderTexture(m_Resolution, m_Resolution, 0);
        rt.filterMode = FilterMode.Point;
        rt.wrapMode = TextureWrapMode.Clamp;
        rt.enableRandomWrite = true;
        
        return rt;
    }

    void ClearTexture()
    {
        RenderTexture texture = new RenderTexture(m_Resolution, m_Resolution, 0);
        texture.filterMode = FilterMode.Point;
        texture.wrapMode = TextureWrapMode.Clamp;
        texture.enableRandomWrite = true;

        Graphics.Blit(texture, m_DrawTexture);
        texture.Release();
    }
}
