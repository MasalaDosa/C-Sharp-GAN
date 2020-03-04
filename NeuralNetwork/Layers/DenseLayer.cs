using System;
using MatrixLib;
using NeuralNetwork.Layers.Activations;

namespace NeuralNetwork.Layers
{
    public class DenseLayer : Layer
    {
        public int InputDim { get; }

        public int OutputNeurons { get; }

        public Activation? Activation { get; }

        public DenseLayer(int inputDim, int outputNeurons, Activation? act, string name = "dense") : base(name)
        {
            // glorot_uniform distribution
            double range = Math.Sqrt(6d / (inputDim + outputNeurons));
            Parameters["weights"] = Matrix.UniformRandomised(-range, range, inputDim, outputNeurons);

            InputDim = inputDim;
            OutputNeurons = outputNeurons;
            Activation = act;
        }

        public override void Forward(Matrix input)
        {
            base.Forward(input); // stores the input.

            // Apply the weights to the input
            Output = input.MatrixMultiply(Parameters["weights"]);
            // And apply the activation if present.
            if (Activation != null)
            {
                Activation.Forward(Output);
                Output = Activation.Output;
            }
        }

        public override void Backward(Matrix gradient)
        {
            if (Activation != null)
            {
                Activation.Backward(gradient);
                gradient = Activation.InputGradient!;
            }
          
            // Gets passed to the previous layer as it's gradient
            // This is essentially the loss being passed backwards through the layers.
            InputGradient = gradient.MatrixMultiply(Parameters["weights"].Transpose());

            // Used by the optimsers - these are the inputs multiplied by the current gradient
            // and represent the slope of the loss function wrt that particular weight
            Gradients["weights"] = Input!.Transpose().MatrixMultiply(gradient);
        }
    }
}
