using System;
using System.Collections.Generic;
using System.Linq;
using MatrixLib;
using NeuralNetwork.Layers;

namespace NeuralNetwork.Optimisers
{
    public class AdamOptimiser : Optimiser
    {
        private Dictionary<string, Matrix> ms;
        private Dictionary<string, Matrix> vs;
        private long iteration;

        public AdamOptimiser(double learningRate = 0.01, double beta1 = 0.9, double beta2 = 0.999) : base("adam", learningRate)
        {
            Beta1 = beta1;
            Beta2 = beta2;
            ms = new Dictionary<string, Matrix>();
            vs = new Dictionary<string, Matrix>();
            iteration = 0;
        }

        public double Beta1 { get; }

        public double Beta2 { get;}

        public override void Update(Layer layer)
        {
            // Update the iteration every call.
            // t <- t + 1
            iteration++;

            //Loop through all the parameters in the layer
            foreach (var p in layer.Parameters.ToList())
            {
                string paramFullName = $"{layer.Name}_{layer.Guid}_{p.Key}";

                //Get the weight values
                Matrix weights = p.Value;

                //Get the gradient/partial derivative values
                Matrix grad = layer.Gradients[p.Key];

                //If this is first time, initilise all the moving average values with 0
                if (!ms.ContainsKey(paramFullName))
                {
                    ms[paramFullName] = Matrix.Zeroes(weights.Rows, weights.Columns);
                    vs[paramFullName] = Matrix.Zeroes(weights.Rows, weights.Columns);
                }

                // Calculate the exponential moving average for Beta 1 against the gradient
                // m_t <- beta1 * m_{ t - 1}
                //                +(1 - beta1) * gradient
                ms[paramFullName] = Matrix.ApplyElementwiseFunction(ms[paramFullName], grad,
                    (m, g) =>
                    Beta1 * m + (1 - Beta1) * g
                );

                //Calculate the exponential squared moving average for Beta 2 against the gradient
                // v_t <- beta2 * v_{ t - 1}
                //                +(1 - beta2) * gradient * *2
                vs[paramFullName] = Matrix.ApplyElementwiseFunction(vs[paramFullName], grad,
                    (v, g) =>
                    Beta2 * v + (1 - Beta2) * Math.Pow(g, 2)
                );

                //lr_t <- learning_rate * sqrt(1 - beta2 ^ t) / (1 - beta1 ^ t)
                double learningRateForThisIteration = LearningRate * Math.Sqrt(1 - Math.Pow(Beta2, iteration)) / (1 - Math.Pow(Beta1, iteration));
                //Console.WriteLine($"LearningRate {LearningRate} Iteration {iteration} LRI {learningRateForThisIteration}");

                //variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
                layer.Parameters[p.Key] = Matrix.ApplyElementwiseFunction(weights, ms[paramFullName], vs[paramFullName],
                    (w, m, v) =>
                    w - (learningRateForThisIteration * m / (Math.Sqrt(v) + Model.Epsilon))
                );
            }
        }
    }
}
