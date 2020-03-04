using System;
using MatrixLib;

namespace NeuralNetwork.Costs
{
    public class BinaryCrossEntropyCost : Cost
    {
        public BinaryCrossEntropyCost(string name = "binary cross entropy") : base(name) { }

        public override Matrix Forward(Matrix predictions, Matrix labels)
        {
            var min = Model.Epsilon;
            var max = 1 - Model.Epsilon;

            // cost = -Sigma(label * Log(prediction) + (1 - label) * Log(1 - prediction))/N
            var clipped = Matrix.ApplyElementwiseFunction(predictions,
                p =>
                p < min ? min : p > max ? max : p
            );

            var output = Matrix.ApplyElementwiseFunction(clipped, labels,
                (c, l) =>
                (-(l * Math.Log(c) + (1 - l) * Math.Log(1 - c)))
            ).
            Average();

            return output;
        }

        public override Matrix Backward(Matrix predictions, Matrix labels)
        {
            // dcost/dprediction = (prediction - label) / (prediction * (1 - prediction))
            return Matrix.ApplyElementwiseFunction(predictions, labels,
                (p, l) =>
                (p - l) / (p * (1 - p))
            );
        }
    }
}
