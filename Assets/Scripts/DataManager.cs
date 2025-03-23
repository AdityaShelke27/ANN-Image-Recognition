using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

public class DataManager : MonoBehaviour
{
    [SerializeField] static string trainImagesPath;
    [SerializeField] static string testImagesPath;
    [SerializeField] static string trainLabelsPath;
    [SerializeField] static string testLabelsPath;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        //ConvertData(trainImagesPath, trainLabelsPath, "TrainImages", "TrainLabels");
        //ConvertData(testImagesPath, testLabelsPath, "TestImages", "TestLabels");
        //LoadData("/ProperDataset/TrainImages", "/ProperDataset/TrainLabels", 100);
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public static void ConvertData(string imagePath, string labelPath, string imageName, string labelName)
    {
        byte[] imageArr = File.ReadAllBytes(Application.dataPath + imagePath);
        byte[] labelArr = File.ReadAllBytes(Application.dataPath + labelPath);

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

            List<byte> imageBytes = new();
            List<byte> labelBytes = new();
            for (int i = 0; i < numberOfImages; i++)
            {
                byte label = labelReader.ReadByte();
                labelBytes.Add(label);
                byte[][] image = new byte[rows][];
                for(int count = 0; count < rows; count++)
                {
                    image[count] = imgReader.ReadBytes(cols);
                }
                Array.Reverse(image);
                //byte[] imageSingle = new byte[rows * cols];
                //int counter = 0;
                for(int j = 0; j < image.Length; j++)
                {
                    for(int k = 0; k < image[j].Length; k++)
                    {
                        imageBytes.Add(image[j][k]);
                        //imageSingle[counter] = image[j][k];
                        //counter++;
                    }
                }
                /*Debug.Log($"Image {i}: Label={label}");
                Texture2D texture = CreateTexture(imageSingle, rows, cols);
                SaveTexture(texture, $"Image {i}");*/
            }
            SaveProperImageData(magicNumberImages, numberOfImages, rows, cols, imageBytes.ToArray(), imageName);
            SaveProperLabelData(magicNumberImages, numberOfImages, labelBytes.ToArray(), labelName);
        }

        Debug.Log("Done");
    }
    public static void LoadAndSaveImage(string imagePath, string labelPath, int amount = 1)
    {
        byte[] imageArr = File.ReadAllBytes(Application.dataPath + imagePath);
        byte[] labelArr = File.ReadAllBytes(Application.dataPath + labelPath);

        using (BinaryReader imgReader = new BinaryReader(new MemoryStream(imageArr)))
        using (BinaryReader labelReader = new BinaryReader(new MemoryStream(labelArr)))
        {
            int magicNumberImages = imgReader.ReadInt32();
            int numberOfImages = imgReader.ReadInt32();
            int rows = imgReader.ReadInt32();
            int cols = imgReader.ReadInt32();

            int magicNumberLabels = labelReader.ReadInt32();
            int numberOfLabels = labelReader.ReadInt32();

            Debug.Log($"Images: {numberOfImages}, Labels: {numberOfLabels}, Size: {rows}x{cols}");
            int imagesToLoad = amount == 0 ? numberOfImages : amount;
            for (int i = 0; i < imagesToLoad; i++)
            {
                byte label = labelReader.ReadByte();
                byte[] image = imgReader.ReadBytes(rows * cols);

                Debug.Log($"Image {i}: Label={label}");
                Texture2D texture = CreateTexture(image, rows, cols);
                SaveTexture(texture, $"Image {i}");
            }
        }
    }
    public static (int imagesToLoad, int rows, int cols, byte[][] images) LoadImages(string imagePath, int amount)
    {
        byte[] imageArr = File.ReadAllBytes(Application.dataPath + imagePath);

        using BinaryReader imgReader = new BinaryReader(new MemoryStream(imageArr));
        int magicNumberImages = imgReader.ReadInt32();
        int numberOfImages = imgReader.ReadInt32();
        int rows = imgReader.ReadInt32();
        int cols = imgReader.ReadInt32();
        int imagesToLoad = amount == 0 ? numberOfImages : amount;
        byte[][] images = new byte[imagesToLoad][];

        for (int i = 0; i < imagesToLoad; i++)
        {
            images[i] = imgReader.ReadBytes(rows * cols);
        }
        imgReader.Close();
        return (imagesToLoad, rows, cols, images);
    }
    public static (int labelsToLoad, byte[] labels) LoadLabels(string labelPath, int amount)
    {
        byte[] labelArr = File.ReadAllBytes(Application.dataPath + labelPath);

        using BinaryReader labelReader = new BinaryReader(new MemoryStream(labelArr));
        int magicNumberLabels = labelReader.ReadInt32();
        int numberOfLabels = labelReader.ReadInt32();
        int labelsToLoad = amount == 0 ? numberOfLabels : amount;
        byte[] labels = new byte[labelsToLoad];

        for (int i = 0; i < labelsToLoad; i++)
        {
            labels[i] = labelReader.ReadByte();
        }
        labelReader.Close();
        return (labelsToLoad, labels);
    }
    public static int ReverseInt(int value) => BitConverter.ToInt32(BitConverter.GetBytes(value).Reverse().ToArray(), 0);
    public static Texture2D CreateTexture(byte[] pixels, int width, int height)
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
    public static void SaveTexture(Texture2D texture, string fileName)
    {
        byte[] pngData = texture.EncodeToPNG();
        File.WriteAllBytes(Application.dataPath + $"/Images/{fileName}.png", pngData);
        Debug.Log($"Save {fileName}");
    }
    public static void SaveProperImageData(int magicNumberImages, int numberOfImages, int rows, int cols, byte[] images, string name)
    {
        List<byte> allBytes = new();
        byte[] arr = BitConverter.GetBytes(magicNumberImages);
        for(int i = 0; i < arr.Length; i++)
        {
            allBytes.Add(arr[i]);
        }
        arr = BitConverter.GetBytes(numberOfImages);
        for (int i = 0; i < arr.Length; i++)
        {
            allBytes.Add(arr[i]);
        }
        arr = BitConverter.GetBytes(rows);
        for (int i = 0; i < arr.Length; i++)
        {
            allBytes.Add(arr[i]);
        }
        arr = BitConverter.GetBytes(cols);
        for (int i = 0; i < arr.Length; i++)
        {
            allBytes.Add(arr[i]);
        }
        for (int i = 0; i < images.Length; i++)
        {
            allBytes.Add(images[i]);
        }

        arr = allBytes.ToArray();
        File.WriteAllBytes(Application.dataPath + $"/{name}", arr);
    }
    public static void SaveProperLabelData(int magicNumberLabels, int numberOfLabels, byte[] labels, string name)
    {
        List<byte> allBytes = new();
        byte[] arr = BitConverter.GetBytes(magicNumberLabels);
        for (int i = 0; i < arr.Length; i++)
        {
            allBytes.Add(arr[i]);
        }
        arr = BitConverter.GetBytes(numberOfLabels);
        for (int i = 0; i < arr.Length; i++)
        {
            allBytes.Add(arr[i]);
        }
        for (int i = 0; i < labels.Length; i++)
        {
            allBytes.Add(labels[i]);
        }

        arr = allBytes.ToArray();
        File.WriteAllBytes(Application.dataPath + $"/{name}", arr);
    }
}
