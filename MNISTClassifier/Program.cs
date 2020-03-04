using System.Linq;
using MatrixLib;
using MNIST;
using NeuralNetwork;
using NeuralNetwork.Costs;
using NeuralNetwork.Layers;
using NeuralNetwork.Layers.Activations;
using NeuralNetwork.Optimisers;
using Utility.UI;

namespace MNISTClassifier
{
    class Program
    {
        static void Main(string[] args)
        {
            ConsoleUI.WriteLine("This program builds, trains, and evaluates several different MNIST Classifiers.");
            ConsoleUI.WriteLine();
            var choice = ConsoleUI.ShowMenu(
                "Please choose one of the options below.",
                "Two Sigmoid, Mean Squared Error, and Stocastic Gradient Descent.",
                "LeakyReLU, Sigmoid, Mean Squared Error, and Stocastic Gradient Descent.",
                "TanH, Sigmoid, Mean Squared Error, and Stocastic Gradient Descent.",
                "Two Sigmoid, Mean Squared Error, and Adam.",
                "LeakyReLU, Sigmoid, Mean Squared Error, and Adam.",
                "TanH, Sigmoid, Mean Squared Error, and Adam."

                );
            switch (choice)
            {
                case 1:
                    MNISTTwoSigmoidSDGAndMeanSquaredError();
                    break;
                case 2:
                    MNISTLeakyReLUSigmoidSDGAndMeanSquareError();
                    break;
                case 3:
                    MNISTLTanHSigmoidSDGAndMeanSquareError();
                    break;
                case 4:
                    MNISTTwoSigmoidAdamAndMeanSquaredError();
                    break;
                case 5:
                    MNISTLeakyReLUSigmoidAdamAndMeanSquaredError();
                    break;
                default:
                    MNISTTanHSigmoidAdamAndMeanSquaredError();
                    break;
            }
        }

        static void MNISTTwoSigmoidSDGAndMeanSquaredError()
        {
            ConsoleUI.WriteLine("Sigmoid, Sigmoid - MSE, SGD");

            var (XTrain, yTrain) = MNISTHelper.LoadTraining(0, 1);
            var (XTest, yTest) = MNISTHelper.LoadTesting(0, 1);

            var model = new Model(new SGDOptimiser(learningRate: 0.1), new MeanSquaredErrorCost());
            model.Add(new DenseLayer(28 * 28, 100, new SigmoidActivation()));
            model.Add(new DenseLayer(100, 10, new SigmoidActivation()));

            MNISTTest(model, XTrain, yTrain, XTest, yTest, 1);
        }

        static void MNISTLeakyReLUSigmoidSDGAndMeanSquareError()
        {
            ConsoleUI.WriteLine("LeakyReLU, Sigmoid - MSE, SGD");

            var (XTrain, yTrain) = MNISTHelper.LoadTraining(0, 1);
            var (XTest, yTest) = MNISTHelper.LoadTesting(0, 1);

            var model = new Model(new SGDOptimiser(learningRate: 0.1), new MeanSquaredErrorCost());
            model.Add(new DenseLayer(28 * 28, 100, new LeakyReLUActivation(0.05)));
            model.Add(new DenseLayer(100, 10, new SigmoidActivation()));

            MNISTTest(model, XTrain, yTrain, XTest, yTest, 1);
        }

        static void MNISTLTanHSigmoidSDGAndMeanSquareError()
        {
            ConsoleUI.WriteLine("TanH, Sigmoid - MSE, SGD");

            var (XTrain, yTrain) = MNISTHelper.LoadTraining(-1, 1);
            var (XTest, yTest) = MNISTHelper.LoadTesting(-1, 1);

            var model = new Model(new SGDOptimiser(learningRate: 0.1), new MeanSquaredErrorCost());
            model.Add(new DenseLayer(28 * 28, 100, new TanhActivation()));
            model.Add(new DenseLayer(100, 10, new SigmoidActivation()));

            MNISTTest(model, XTrain, yTrain, XTest, yTest, 1);
        }

        static void MNISTTwoSigmoidAdamAndMeanSquaredError()
        {
            ConsoleUI.WriteLine("Sigmoid, Sigmoid - MSE, Adam");

            var (XTrain, yTrain) = MNISTHelper.LoadTraining(0, 1);
            var (XTest, yTest) = MNISTHelper.LoadTesting(0, 1);

            var model = new Model(new AdamOptimiser(learningRate: 2e-2, beta1: 0.5), new MeanSquaredErrorCost());
            model.Add(new DenseLayer(28 * 28, 100, new SigmoidActivation()));
            model.Add(new DenseLayer(100, 10, new SigmoidActivation()));

            MNISTTest(model, XTrain, yTrain, XTest, yTest, 1);
        }

        static void MNISTLeakyReLUSigmoidAdamAndMeanSquaredError()
        {
            ConsoleUI.WriteLine("LeakyReLU, Sigmoid - MSE, Adam");

            var (XTrain, yTrain) = MNISTHelper.LoadTraining(0, 1);
            var (XTest, yTest) = MNISTHelper.LoadTesting(0, 1);

            var model = new Model(new AdamOptimiser(learningRate: 2e-3, beta1: 0.5), new MeanSquaredErrorCost());
            model.Add(new DenseLayer(28 * 28, 100, new LeakyReLUActivation(0.05)));
            model.Add(new DenseLayer(100, 10, new SigmoidActivation()));

            MNISTTest(model, XTrain, yTrain, XTest, yTest, 1);
        }

        static void MNISTTanHSigmoidAdamAndMeanSquaredError()
        {
            ConsoleUI.WriteLine("TanH, Sigmoid - MSE, Adam");

            var (XTrain, yTrain) = MNISTHelper.LoadTraining(-1, 1);
            var (XTest, yTest) = MNISTHelper.LoadTesting(-1, 1);

            var model = new Model(new AdamOptimiser(learningRate: 2e-3, beta1: 0.5), new MeanSquaredErrorCost());
            model.Add(new DenseLayer(28 * 28, 100, new TanhActivation()));
            model.Add(new DenseLayer(100, 10, new SigmoidActivation()));

            MNISTTest(model, XTrain, yTrain, XTest, yTest, 1);
        }

        /// <summary>
        /// Train and evaluate the supplied MNIST Model.
        /// </summary>
        /// <param name="model"></param>
        /// <param name="XTrain"></param>
        /// <param name="yTrain"></param>
        /// <param name="XTest"></param>
        /// <param name="yTest"></param>
        /// <param name="numEpochs"></param>
        private static void MNISTTest(Model model, Matrix XTrain, Matrix yTrain, Matrix XTest, Matrix yTest, int numEpochs = 1)
        {
            ConsoleUI.WriteLine("Training...");
            model.BatchEnd += Model_BatchEnd;
            model.EpochEnd += Model_EpochEnd;
            model.Train(XTrain, yTrain, numEpochs, 20);
            model.BatchEnd -= Model_BatchEnd;
            model.EpochEnd -= Model_EpochEnd;
            ConsoleUI.WriteLine("Completed.");

            ConsoleUI.WriteLine("Evaluating...");
            int testCount = 0, testPositive = 0;
            int currentIndex = 0;

            while (XTest.CanSliceRows(currentIndex, 1))
            {
                var data = XTest.SliceRows(currentIndex, 1);
                var label = yTest.SliceRows(currentIndex, 1);
                int actualLabel = label.Data.ToList().IndexOf(label.Data.Max());
                var prediction = model.Predict(data);
                var predictedLabel = prediction.Data.ToList().IndexOf(prediction.Data.Max());

                if (actualLabel == predictedLabel)
                {
                    testPositive += 1;
                }
                testCount += 1;

                currentIndex += 1;
            }

            ConsoleUI.WriteLine($"Completed: {(double)testPositive / (double)testCount}.");
        }

        private static void Model_BatchEnd(object? sender, BatchEndEventArgs e)
        {
            if (e.Batch % 25 == 0)
                ConsoleUI.WriteLine(e.ToString());
        }

        private static void Model_EpochEnd(object? sender, EpochEndEventArgs e)
        {
            ConsoleUI.WriteLine(e.ToString());
        }
    }
}
