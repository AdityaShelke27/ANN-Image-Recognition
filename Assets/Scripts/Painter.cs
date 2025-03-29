using UnityEngine;
using UnityEngine.UI;

public class Painter : MonoBehaviour
{
    [SerializeField] BoxCollider2D m_CanvasCollider;
    [SerializeField] int m_Height;
    [SerializeField] int m_Width;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        m_CanvasCollider.GetComponent<MeshRenderer>().material.mainTexture = CreateTexture();
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 pos = Camera.main.ScreenToWorldPoint(Input.mousePosition);
    }

    Texture2D CreateTexture()
    {
        Texture2D rt = new Texture2D(m_Height, m_Width);
        
        return rt;
    }
}
