using System;
using System.Collections.Generic;
using System.Linq;
using MatrixLib;
using NeuralNetwork.Costs;
using NeuralNetwork.Layers;
using NeuralNetwork.Optimisers;

namespace NeuralNetwork
{
    public class Model
    {
        public const double Epsilon = 1e-7;

        public Model(Optimiser optimiser, Cost cost)
        {
            Layers = new List<Layer>();
            Optimiser = optimiser;
            Cost = cost;
        }

        public event EventHandler<BatchEndEventArgs>? BatchEnd;

        public event EventHandler<EpochEndEventArgs>? EpochEnd;

        public List<Layer> Layers { get; }

        public Optimiser Optimiser { get; }

        public Cost Cost { get; }

        public List<double> TrainingLoss { get; set; } = new List<double>();

        public void Add(Layer layer)
        {
            Layers.Add(layer);
        }

        /// <summary>
        /// Run the network forwards given a matrix representing the inputs
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Matrix Forward(Matrix input)
        {
            Layer? previousLayer = null;
            foreach (var layer in Layers)
            {
                if (previousLayer == null)
                    layer.Forward(input);
                else
                    layer.Forward(previousLayer.Output!);
                previousLayer = layer;
            }
            return previousLayer!.Output!;
        }

        /// <summary>
        /// Run the network backwards given a matrix representing the gradient of the cost
        /// </summary>
        /// <param name="gradOutput"></param>
        public void Backward(Matrix gradOutput)
        {
            var curGradOutput = gradOutput;
            for (int i = Layers.Count - 1; i >= 0; --i)
            {
                var layer = Layers[i];
                layer.Backward(curGradOutput);
                curGradOutput = layer.InputGradient!;
            }
        }

        public Matrix Predict(Matrix inputs)
        {
            return Forward(inputs);
        }

        public void Train(Matrix trainingData, Matrix labels, int epochs, int batchSize)
        {
            List<double> batchLoss = new List<double>();

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                var epochLossAvg = TrainBatches(trainingData, labels, batchSize, epoch);

                TrainingLoss.Add(epochLossAvg);

                EpochEndEventArgs eventArgs = new EpochEndEventArgs(epoch, epochLossAvg);
                EpochEnd?.Invoke(epoch, eventArgs);
            }
        }

        private double TrainBatches(Matrix trainingData, Matrix labels, int batchSize, int epoch)
        {
            int currentIndex = 0;
            int currentBatch = 1;
            List<double> batchLosses = new List<double>(); ;

            //Loop untill the data is exhauted for every batch selected
            while (trainingData.CanSliceRows(currentIndex, batchSize))
            {
                //Get the batch data based on the specified batch size
                var xtrain = trainingData.SliceRows(currentIndex, batchSize);
                var ytrain = labels.SliceRows(currentIndex, batchSize);

                //Run forward for all the layers to predict the value for the training set
                var ypred = Forward(xtrain);

                //Find the loss/cost value for the prediction wrt expected result
                var costVal = Cost.Forward(ypred, ytrain);
                batchLosses.Add(costVal.Data[0]);

                //Get the gradient of the cost function which is then passed to the layers during back-propagation
                var grad = Cost.Backward(ypred, ytrain);
                //Run back-propagation accross all the layers
                Backward(grad);
                //Now time to update the neural network weights using the specified optimizer function
                foreach (var layer in Layers)
                {
                    Optimiser.Update(layer);
                }
                currentIndex = currentIndex + batchSize;
                double batchLossAvg = Math.Round(costVal.Data[0], 3);

                BatchEndEventArgs eventArgs1 = new BatchEndEventArgs(epoch, currentBatch, batchLossAvg);
                BatchEnd?.Invoke(epoch, eventArgs1);
                currentBatch += 1;
            }

            return Math.Round(batchLosses.Average(), 3);
        }
    }
}
