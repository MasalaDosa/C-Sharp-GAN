using System.Linq;
using MatrixLib;
using NeuralNetwork.Layers;

namespace NeuralNetwork.Optimisers
{
    public class SGDOptimiser : Optimiser
    {
        public SGDOptimiser(double learningRate = 0.01) : base("stochastic gradient descent", learningRate){ }

        public override void Update(Layer layer)
        {
            //Loop through all the parameters in the layer
            foreach (var p in layer.Parameters.ToList())
            {
                //Get the parameter name
                string paramName = p.Key;

                //Get the gradient/partial derivative values
                Matrix gradient = layer.Gradients[p.Key];

                //Update the weight of of the neurons
                layer.Parameters[p.Key] = Matrix.ApplyElementwiseFunction(p.Value, gradient,
                    (w, g) =>
                    w - (LearningRate * g)
                );
            }
        }
    }
}
