using System.Collections.Generic;
using UnityEngine;

public static class ImageProcessor
{
    public static double[] BlackWhiteImage(double[] image, float threshold)
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
    public static float[] BlackWhiteImage(float[] image, float threshold)
    {
        float[] newImage = new float[image.Length];

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
    public static float[] KerneledImage(float[] image, float[] kernel)
    {
        int imageLength = (int)Mathf.Sqrt(image.Length);
        int kernelLength = (int)Mathf.Sqrt(kernel.Length);

        float[][] image2d = new float[imageLength][];
        float[][] kernel2d = new float[kernelLength][];

        int count = 0;
        for (int i = 0; i < imageLength; i++)
        {
            image2d[i] = new float[imageLength];
            for (int j = 0; j < imageLength; j++)
            {
                image2d[i][j] = image[count];
                count++;
            }
        }
        count = 0;
        for (int i = 0; i < kernelLength; i++)
        {
            kernel2d[i] = new float[kernelLength];
            for (int j = 0; j < kernelLength; j++)
            {
                kernel2d[i][j] = kernel[count];
                count++;
            }
        }
        int clampVal = imageLength - kernelLength + 1;
        float[] newImage = new float[(int)Mathf.Pow(clampVal, 2)];
        count = 0;
        for (int i = 0; i < clampVal; i++)
        {
            for (int j = 0; j < clampVal; j++)
            {
                float sum = 0;
                for (int k = 0; k < kernelLength; k++)
                {
                    for (int l = 0; l < kernelLength; l++)
                    {
                        sum += image2d[i + k][j + l] * kernel2d[k][l];
                    }
                }
                newImage[count] = sum;
                count++;
            }
        }

        return newImage;
    }
    public static double[] KerneledImage(double[] image, double[] kernel)
    {
        int imageLength = (int)Mathf.Sqrt(image.Length);
        int kernelLength = (int)Mathf.Sqrt(kernel.Length);

        double[][] image2d = new double[imageLength][];
        double[][] kernel2d = new double[kernelLength][];

        int count = 0;
        for (int i = 0; i < imageLength; i++)
        {
            image2d[i] = new double[imageLength];
            for (int j = 0; j < imageLength; j++)
            {
                image2d[i][j] = image[count];
                count++;
            }
        }
        count = 0;
        for (int i = 0; i < kernelLength; i++)
        {
            kernel2d[i] = new double[kernelLength];
            for (int j = 0; j < kernelLength; j++)
            {
                kernel2d[i][j] = kernel[count];
                count++;
            }
        }
        int clampVal = imageLength - kernelLength + 1;
        double[] newImage = new double[(int)Mathf.Pow(clampVal, 2)];
        count = 0;
        for (int i = 0; i < clampVal; i++)
        {
            for (int j = 0; j < clampVal; j++)
            {
                double sum = 0;
                for (int k = 0; k < kernelLength; k++)
                {
                    for (int l = 0; l < kernelLength; l++)
                    {
                        sum += image2d[i + k][j + l] * kernel2d[k][l];
                    }
                }
                newImage[count] = sum;
                count++;
            }
        }

        return newImage;
    }
    public static double[] MaxPool(double[] image, int poolSize)
    {
        int imageLength = (int)Mathf.Sqrt(image.Length);
        double[][] image2d = new double[imageLength][];

        int count = 0;
        for (int i = 0; i < imageLength; i++)
        {
            image2d[i] = new double[imageLength];
            for (int j = 0; j < imageLength; j++)
            {
                image2d[i][j] = image[count];
                count++;
            }
        }

        List<double> poolImage = new();

        for (int i = 0; i < imageLength; i += poolSize)
        {
            for (int j = 0; j < imageLength; j += poolSize)
            {
                double MaxVal = double.MinValue;
                for(int k = 0; k < poolSize && i + k < imageLength; k++)
                {
                    for (int l = 0; l < poolSize && j + l < imageLength; l++)
                    {
                        if (image2d[i + k][j + l] > MaxVal)
                        {
                            MaxVal = image2d[i + k][j + l];
                        }
                    }
                }
                poolImage.Add(MaxVal);
            }
        }

        return poolImage.ToArray();
    }
    public static float[] MaxPool(float[] image, int poolSize)
    {
        int imageLength = (int)Mathf.Sqrt(image.Length);
        float[][] image2d = new float[imageLength][];

        int count = 0;
        for (int i = 0; i < imageLength; i++)
        {
            image2d[i] = new float[imageLength];
            for (int j = 0; j < imageLength; j++)
            {
                image2d[i][j] = image[count];
                count++;
            }
        }

        List<float> poolImage = new();

        for (int i = 0; i < imageLength; i += poolSize)
        {
            for (int j = 0; j < imageLength; j += poolSize)
            {
                float MaxVal = float.MinValue;
                for (int k = 0; k < poolSize && i + k < imageLength; k++)
                {
                    for (int l = 0; l < poolSize && j + l < imageLength; l++)
                    {
                        if (image2d[i + k][j + l] > MaxVal)
                        {
                            MaxVal = image2d[i + k][j + l];
                        }
                    }
                }
                poolImage.Add(MaxVal);
            }
        }

        return poolImage.ToArray();
    }

    public static Texture2D TransformImage(Texture2D original, float rotationDegrees, Vector2 scale, Vector2 position, GameObject tempGO, SpriteRenderer sr, Camera cam, RenderTexture rt)
    {
        sr.sprite = Sprite.Create(original, new Rect(0, 0, original.width, original.height), new Vector2(0.5f, 0.5f));

        // 2. Apply transforms
        tempGO.transform.rotation = Quaternion.Euler(0, 0, rotationDegrees);
        tempGO.transform.localScale = new Vector3(scale.x, scale.y, 1);
        tempGO.transform.position = new Vector3(position.x, position.y, 0);

        
        cam.targetTexture = rt;
        cam.Render();

        // 4. Read pixels into new texture
        RenderTexture.active = rt;
        Texture2D result = new Texture2D(28, 28, TextureFormat.RGBA32, false);
        result.ReadPixels(new Rect(0, 0, 28, 28), 0, 0);
        result.Apply();

        // Cleanup
        RenderTexture.active = null;

        return result;
    }
}
