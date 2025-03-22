using System;
using System.IO;
using System.Linq;
using TMPro;
using UnityEngine;

public class TextTesting : MonoBehaviour
{
    [SerializeField] string imagesPath;
    [SerializeField] string labelsPath;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        LoadData();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void LoadData()
    {
        byte[] imageArr = File.ReadAllBytes(Application.dataPath + imagesPath);
        byte[] labelArr = File.ReadAllBytes(Application.dataPath + labelsPath);

        using (BinaryReader imgReader = new BinaryReader(new MemoryStream(imageArr)))
        using (BinaryReader labelReader = new BinaryReader(new MemoryStream(labelArr)))
        {
            int magicNumberImages = ReverseInt(imgReader.ReadInt32());
            int numberOfImages = ReverseInt(imgReader.ReadInt32());
            int rows = ReverseInt(imgReader.ReadInt32());
            int cols = ReverseInt(imgReader.ReadInt32());

            int magicNumberLabels = ReverseInt(labelReader.ReadInt32());
            int numberOfLabels = ReverseInt(labelReader.ReadInt32());

            Debug.Log($"Images: {numberOfImages}, Labels: {numberOfLabels}, Size: {rows}x{cols}");

            for(int i = 0; i < numberOfImages; i++)
            {
                byte label = labelReader.ReadByte();
                byte[] image = imgReader.ReadBytes(rows * cols);

                Debug.Log($"Image {i}: Label={label}");
                Texture2D texture = CreateTexture(image, rows, cols);
                SaveTexture(texture, $"Image {i}");
            }
        }
    }

    int ReverseInt(int value) => BitConverter.ToInt32(BitConverter.GetBytes(value).Reverse().ToArray(), 0);

    Texture2D CreateTexture(byte[] pixels, int width, int height)
    {
        Texture2D texture = new Texture2D(width, height);
        Color[] colors = new Color[width * height];

        for(int i = 0; i < pixels.Length; i++)
        {
            float intensity = pixels[i] / 255.0f;
            colors[i] = new Color(intensity, intensity, intensity, 1);
        }

        texture.SetPixels(colors);
        texture.Apply();

        return texture;
    }

    void SaveTexture(Texture2D texture, string fileName)
    {
        byte[] pngData = texture.EncodeToPNG();
        File.WriteAllBytes(Application.dataPath + $"/Images/{fileName}.png", pngData);
        Debug.Log($"Save {fileName}");
    }
}
