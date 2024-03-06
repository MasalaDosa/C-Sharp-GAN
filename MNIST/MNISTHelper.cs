using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using MatrixLib;
using Utility.UI;

namespace MNIST
{
    /// <summary>
    /// A helper class for loading the MNIST Dataset.
    /// </summary>
    public static class MNISTHelper
    {
        // If the MNIST dataset cannot be found locally then an attempt will be made to download from these locations
        const string MNIST_TRAIN_URL = "http://www.pjreddie.com/media/files/mnist_train.csv";
        const string MNIST_TEST_URL = "http://www.pjreddie.com/media/files/mnist_test.csv";

        // The local folder to search for MNIST Data.
        const string MNIST_DIRECTORY = "MNIST_DATA";

        // Paths for the local MNIST data
        static string MNISTTrainPath => Path.Combine(MNIST_DIRECTORY, "mnist_train.csv");

        static string MNISTTestPath => Path.Combine(MNIST_DIRECTORY, "mnist_test.csv");

        /// <summary>
        /// Load the MNIST Training data.
        /// </summary>
        /// <param name="scaleMin">Data will be scaled so that this is the minimum value</param>
        /// <param name="scaleMax">Data will be scaled so that this is the maximum value</param>
        /// <param name="displayExample">Optionally show a representation of a random MNIST digit</param>
        /// <param name="filter">If specified load only data that matches this filter</param>
        /// <returns></returns>
        public static (Matrix XTrain, Matrix yTrain) LoadTraining(double scaleMin = 0d, double scaleMax = 1d, int[]? filter = null)
        {
            DownloadMNISTData().GetAwaiter().GetResult();
            var data = MNISTLoad(MNISTTrainPath, scaleMin, scaleMax, filter);
            ConsoleUI.WriteLine($"XTrain: {data.XTrain}");
            ConsoleUI.WriteLine($"yTrain: {data.yTrain}");
            return data;
        }

        /// <summary>
        /// Load the MNIST Testing data.
        /// </summary>
        /// <param name="scaleMin">Data will be scaled so that this is the minimum value</param>
        /// <param name="scaleMax">Data will be scaled so that this is the maximum value</param>
        /// <param name="displayExample">Optionally show a representation of a random MNIST digit</param>
        /// <param name="filter">If specified load only data that matches this filter</param>
        /// <returns></returns>
        public static (Matrix XTest, Matrix yTest) LoadTesting(double scaleMin = 0d, double scaleMax = 1d, int[]? filter = null)
        {
            DownloadMNISTData().GetAwaiter().GetResult();
            var data = MNISTLoad(MNISTTestPath, scaleMin, scaleMax, filter);
            ConsoleUI.WriteLine($"XTest: {data.XTrain}");
            ConsoleUI.WriteLine($"yTest: {data.yTrain}");
            return data;
        }

        /// <summary>
        /// Attempts to download the MNIST data from a known location.
        /// </summary>
        /// <returns></returns>
        private static async Task DownloadMNISTData()
        {
            try
            {
                if (!Directory.Exists(MNIST_DIRECTORY))
                {
                    Directory.CreateDirectory(MNIST_DIRECTORY);
                }

                if (!File.Exists(MNISTTrainPath))
                {
                    ConsoleUI.WriteLine("Downloading MNIST Training data...");
                    await DownloadFile(MNIST_TRAIN_URL, MNISTTrainPath);
                }

                if (!File.Exists(MNISTTestPath))
                {
                    ConsoleUI.WriteLine("Downloading MNIST Testing data...");
                    await DownloadFile(MNIST_TEST_URL, MNISTTestPath);
                }
            }
            catch (Exception ex)
            {
                ConsoleUI.WriteLine($"Unable to download MNIST Training data to {MNISTTrainPath} and {MNISTTestPath}.");
                ConsoleUI.WriteLine($"Reason: {ex.Message}.");
                ConsoleUI.WriteLine($"Please rectify the problem or download manually from:");
                ConsoleUI.WriteLine(MNIST_TRAIN_URL);
                ConsoleUI.WriteLine(MNIST_TEST_URL);
                Environment.Exit(1);
            }
        }

        /// <summary>
        /// Download a file from a URL
        /// </summary>
        /// <param name="url"></param>
        /// <param name="filename"></param>
        /// <returns></returns>
        private static async Task DownloadFile(string url, string filename)
        {
            using (HttpClient client = new HttpClient())
            {
                client.Timeout = TimeSpan.FromMinutes(10);
                HttpResponseMessage response = await client.GetAsync(url);

                // Check that response was successful or throw exception
                response.EnsureSuccessStatusCode();

                // Read response asynchronously and save asynchronously to file
                using (FileStream fileStream = new FileStream(filename, FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    //copy the content from response to filestream
                    await response.Content.CopyToAsync(fileStream);
                }
            }
        }

        /// <summary>
        /// Load an MNist file into a data matrix [count by 784] and a label matrix [count by 10]
        /// </summary>
        /// <param name="fileName"></param>
        /// <param name="scaleMin">Data will be scaled so that this is the minimum value</param>
        /// <param name="scaleMax">Data will be scaled so that this is the maximum value</param>
        /// <param name="displayExample">Optional display a random MNIST digit</param>
        /// <param name="filter">If specified load only images matching this filter</param>
        /// <returns></returns>
        private static (Matrix XTrain, Matrix yTrain) MNISTLoad(string fileName, double scaleMin, double scaleMax, int[]? filter)
        {
            List<double> labels = new List<double>();
            List<double> training = new List<double>();

            int i = 0;
            foreach (var line in File.ReadLines(fileName))
            {
                var lineData = line.Split(',');

                if (filter == null || !filter.Any() || filter.Contains(int.Parse(lineData[0])))
                {
                    // Set the targets.
                    double[] targets = new double[10];
                    targets = targets.Select(d => 0d).ToArray();

                    targets[int.Parse(lineData[0])] = 1;

                    double[] inputs = new double[784];
                    for (int j = 1; j < lineData.Length; j++)
                    {
                        // Store a scaled version of the inputs
                        inputs[j - 1] = double.Parse(lineData[j]) / 255;
                        inputs[j - 1] = inputs[j - 1] * (scaleMax - scaleMin) + scaleMin;
                    }

                    labels.AddRange(targets);
                    training.AddRange(inputs);
                    i++;
                }
            }

            ConsoleUI.WriteLine($"Loaded {i} images in total");

            var allLabels = Matrix.FromData(labels.ToArray(), i, 10);

            var allData = Matrix.FromData(training.ToArray(), i, 28 * 28);

            return (allData, allLabels);
        }
    }
}
