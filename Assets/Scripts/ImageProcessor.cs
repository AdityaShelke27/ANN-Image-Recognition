using System.Collections.Generic;
using System.Drawing;
using UnityEngine;
using static UnityEngine.UI.Image;

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

    public static float[] TransformTexture(float[] inputTex, float rotationDegrees, Vector2 scale, Vector2 offset, int outputSize = 28)
    {/*
        //Texture2D outputTex = new Texture2D(outputSize, outputSize, TextureFormat.RGBA32, false);
        float[] outputPixels = new float[outputSize * outputSize];

        // Inverse transform matrix
        float angleRad = -rotationDegrees * Mathf.Deg2Rad;
        float cos = Mathf.Cos(angleRad);
        float sin = Mathf.Sin(angleRad);

        float scaleX = 1f / scale.x;
        float scaleY = 1f / scale.y;

        Vector2 center = new Vector2(outputSize / 2f, outputSize / 2f);

        for (int y = 0; y < outputSize; y++)
        {
            for (int x = 0; x < outputSize; x++)
            {
                // Normalize pixel to [-1,1]
                Vector2 p = new Vector2(x, y) - center;

                // Apply inverse transform: scale -> rotate -> translate
                float u = (p.x * cos - p.y * sin) * scaleX - offset.x * outputSize + outputSize / 2f;
                float v = (p.x * sin + p.y * cos) * scaleY - offset.y * outputSize + outputSize / 2f;

                // Bilinear interpolation
                outputPixels[y * outputSize + x] = inputTex[(int)v * outputSize + (int)u];
            }
        }

        return outputPixels;*/
        float[] outputPixels = new float[outputSize * outputSize];
        if (scale.magnitude != 0)
        {
            Vector2 iHat = new Vector2(Mathf.Cos(rotationDegrees), Mathf.Sin(rotationDegrees)) / scale.x;
            Vector2 jHat = new Vector2(-iHat.y, iHat.x);
            for (int y = 0; y < outputSize; y++)
            {
                for (int x = 0; x < outputSize; x++)
                {
                    double u = x / (outputSize - 1.0);
                    double v = y / (outputSize - 1.0);

                    double uTransformed = iHat.x * (u - 0.5) + jHat.x * (v - 0.5) + 0.5 - offset.x;
                    double vTransformed = iHat.y * (u - 0.5) + jHat.y * (v - 0.5) + 0.5 - offset.y;
                    outputPixels[y * outputSize + x] = System.Math.Clamp(inputTex[(int)vTransformed * outputSize + (int)uTransformed], 0, 1);
                }
            }
        }

        return outputPixels;
    }
    public static double[] TransformTexture(double[] inputTex, float rotationDegrees, Vector2 scale, Vector2 offset, int outputSize = 28)
    {
        //Texture2D outputTex = new Texture2D(outputSize, outputSize, TextureFormat.RGBA32, false);
        double[] outputPixels = new double[outputSize * outputSize];

        // Inverse transform matrix
        float angleRad = -rotationDegrees * Mathf.Deg2Rad;
        float cos = Mathf.Cos(angleRad);
        float sin = Mathf.Sin(angleRad);

        float scaleX = 1f / scale.x;
        float scaleY = 1f / scale.y;

        Vector2 center = new Vector2(outputSize / 2f, outputSize / 2f);

        for (int y = 0; y < outputSize; y++)
        {
            for (int x = 0; x < outputSize; x++)
            {
                // Normalize pixel to [-1,1]
                Vector2 p = new Vector2(x, y);// - center;

                // Apply inverse transform: scale -> rotate -> translate
                float u = (p.x * cos - p.y * sin) * scaleX - offset.x * outputSize + outputSize / 2f;
                float v = (p.x * sin + p.y * cos) * scaleY - offset.y * outputSize + outputSize / 2f;

                // Bilinear interpolation
                outputPixels[y * outputSize + x] = inputTex[(int)v * outputSize + (int)u];
            }
        }

        return outputPixels;
    }
}
