using System;
using System.Collections.Generic;
using System.IO;
using MatrixLib;
using MNIST;
using NeuralNetwork;
using NeuralNetwork.Costs;
using NeuralNetwork.Layers;
using NeuralNetwork.Layers.Activations;
using NeuralNetwork.Optimisers;
using Utility.PseudoRandom;
using Utility.UI;

namespace CSharpGAN
{
    class Program
    {
        private const int NUM_EPOCHS = 500;
        private const int BATCH_SIZE = 256;
        private const int NOISE_DIM = 100;
 
        static void Main(string[] args)
        {
            SimpleGAN();
        }

        private static void SimpleGAN()
        {
            string imageFolder = CreateImageFolder();

            ConsoleUI.WriteLine("Enter a comma separated list of numbers from 0 to 9 to train the GAN on those digits.");
            ConsoleUI.WriteLine("Just press enter to train the GAN on all digits (this will take a *long* time).");
            ConsoleUI.WriteLine($"Example generated images will be written into '{imageFolder}' at the end of each epoch.");

            var (XTrain, _) = MNISTHelper.LoadTraining(scaleMin: -1, scaleMax: 1, filter: GetFilterFromUser());

            var adam = new AdamOptimiser(learningRate: 2e-4, beta1: 0.5, beta2: 0.999);
            var generator = BuildGenerator(adam);
            var discriminator = BuildDiscriminator(adam);
            Train(imageFolder, XTrain, generator, discriminator);
        }

        private static string CreateImageFolder()
        {
            var imageFolder = DateTime.Now.ToString("yyyy-MM-dd-HH-mm-ss");
            if (!Directory.Exists(imageFolder))
            {
                Directory.CreateDirectory(imageFolder);
            }

            return imageFolder;
        }

        private static int[] GetFilterFromUser()
        {
            List<int> ints = new List<int>();
            foreach (var digit in ConsoleUI.PromptForListOfIntegers(0, 9))
            {
                ints.Add(digit);
            }
            return ints.ToArray();
        }

        private static Model BuildGenerator(Optimiser optimiser)
        {
            var generator = new Model(optimiser, new BinaryCrossEntropyCost());
            generator.Add(new DenseLayer(NOISE_DIM, 256, new LeakyReLUActivation(0.2)));
            generator.Add(new DenseLayer(256, 512, new LeakyReLUActivation(0.2)));
            generator.Add(new DenseLayer(512, 1024, new LeakyReLUActivation(0.2)));
            generator.Add(new DenseLayer(1024, 28 * 28, new TanhActivation())); // Centred around 0 like our data
            return generator;
        }

        private static Model BuildDiscriminator(Optimiser optimiser)
        {
            var discriminator = new Model(optimiser, new BinaryCrossEntropyCost());
            discriminator.Add(new DenseLayer(28 * 28, 512, new LeakyReLUActivation(0.2)));
            discriminator.Add(new DenseLayer(512, 256, new LeakyReLUActivation(0.2)));
            discriminator.Add(new DenseLayer(256, 1, new SigmoidActivation()));
            return discriminator;
        }

        private static void Train(string imageFolder, Matrix XTrain, Model generator, Model discriminator)
        {
            int batchCount = (int)Math.Ceiling(XTrain.Rows / (double)BATCH_SIZE);
            int halfBatch = (int)Math.Floor(BATCH_SIZE / 2d);

            for (var epoch = 1; epoch <= NUM_EPOCHS; epoch++)
            {
                ConsoleUI.WriteLine($"Starting Epoch {epoch}");
                // Shuffle the training data.
                XTrain.ShuffleRows();
                ConsoleUI.WriteLine("Shuffled training data...");

                for (var step = 1; step <= batchCount; step++)
                {
                    ConsoleUI.WriteLine($"Step {step} of {batchCount}.");
                    TrainDiscriminator(XTrain, halfBatch, generator, discriminator, epoch, step);
                    TrainGenerator(generator, discriminator, epoch, step);
                }

                GenerateExampleImages(generator, 10, 10, Path.Combine(imageFolder, $"{epoch}.png"));
            }
        }

        private static void TrainDiscriminator(Matrix XTrain, int halfBatch, Model generator, Model discriminator, int epoch, int step)
        {
            // Train real images
            var (realImages, realLabels) = GetRealImages(XTrain, halfBatch);
            // Pass forwards through the discriminator
            var forwardReal = discriminator.Forward(realImages);
            // Apply the cost function
            var realLoss = discriminator.Cost.Forward(forwardReal, realLabels);
            // Backward from cost->discrim.
            discriminator.Backward(discriminator.Cost.Backward(forwardReal, realLabels));
            // Optimise the disciminator
            foreach (var layer in discriminator.Layers)
            {
                discriminator.Optimiser.Update(layer);
            }

            // Now train the fake images
            var (fakeImages, fakeLabels) = GetFakeImages(generator, halfBatch);
            // Pass forwards through the discriminator
            var forwardFake = discriminator.Forward(fakeImages);
            // Apply the cost function
            var fakeLoss = discriminator.Cost.Forward(forwardFake, fakeLabels);
            // Backward from cost->discrim.
            discriminator.Backward(discriminator.Cost.Backward(forwardFake, fakeLabels));
            // Optimise the disciminator
            foreach (var layer in discriminator.Layers)
            {
                discriminator.Optimiser.Update(layer);
            }

            var meanLoss = (realLoss[0] + fakeLoss[0]) / 2d;
            ConsoleUI.WriteLine($"Discriminator Epoch {epoch} Step {step} Loss {meanLoss:N3}");
        }

        private static void TrainGenerator(Model generator, Model discriminator, int epoch, int step)
        {
            var noise = GetNoise(BATCH_SIZE);
            var forgedLabels = Matrix.Ones(BATCH_SIZE, 1);

            // Pass forward through Generator and then Discriminator
            var generatorForward = generator.Forward(noise);
            var combinedForward = discriminator.Forward(generatorForward);
            // Apply loss function.

            var combinedLoss = discriminator.Cost.Forward(combinedForward, forgedLabels);

            // Backward from cost->discrim->gen
            discriminator.Backward(discriminator.Cost.Backward(combinedForward, forgedLabels));
            generator.Backward(discriminator.Layers[0].InputGradient!);
            
            // Optimise the generator
            foreach (var layer in generator.Layers)
            {
                generator.Optimiser.Update(layer);
            }

            ConsoleUI.WriteLine($"Generator Epoch {epoch} Step {step} Loss {combinedLoss[0]:N3}");
        }

        private static (Matrix images, Matrix labels) GetRealImages(Matrix XTrain, int numberOfImages)
        {
            int startIdx = PRNG.Basic.GetUniformInt32(0, XTrain.Rows - numberOfImages);
            var images = XTrain.SliceRows(startIdx, numberOfImages);
            var labels =  Matrix.Filled(0.9, numberOfImages, 1);
            return (images, labels);
        }

        private static (Matrix images, Matrix labels) GetFakeImages(Model generator, int numberOfImages)
        {
            var noise = GetNoise(numberOfImages);
            var images = generator.Predict(noise);
            var labels = Matrix.Zeroes(numberOfImages, 1);
            return (images, labels);
        }

        private static Matrix GetNoise(int numberOfImages)
        {
            return Matrix.NormalRandomised(0, 1, numberOfImages, NOISE_DIM);
        }

        private static void GenerateExampleImages(Model generator, int rows, int cols, string filename)
        {
            var (createdImages, _) = GetFakeImages(generator, rows * cols);
            var width = 30 * cols;
            var height = 30 * rows;
            var data = new double[width,height];
            for (var x = 0; x < cols; x++)
            {
                for (var y = 0; y < rows; y++)
                {
                    // Start position for this sub-image
                    int sx = 30 * x + 1;
                    int sy = 30 * y + 1;
                    var subImage = createdImages.SliceRows(x * rows + y, 1);
                    for (var i = 0; i < 28; i++)
                    {
                        for (var j = 0; j < 28; j++)
                        {
                            var subImageIdx = j * 28 + i;
                            data[sx + i, sy + j] = subImage[subImageIdx];
                            
                        }
                    }
                }
            }
            ImageWriter.GenerateImage(data, filename);
        }
    }
}
