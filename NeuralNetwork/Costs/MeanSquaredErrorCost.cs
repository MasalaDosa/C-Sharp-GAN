using System;
using MatrixLib;

namespace NeuralNetwork.Costs
{
    public class MeanSquaredErrorCost : Cost
    {
        public MeanSquaredErrorCost(string name = "mean squared error") : base(name) { }

        public override Matrix Forward(Matrix predictions, Matrix labels)
        {
            // cost = (predictions - labels)^2/N
            var errorSquared = Matrix.ApplyElementwiseFunction(predictions, labels,
                (p, l) =>
                Math.Pow(p - l, 2)
            );
            return errorSquared.Average();
        }

        public override Matrix Backward(Matrix predictions, Matrix labels)
        {
            // dcost/dpredictions =  2 * (predictions - labels) / N
            double norm = 2.0 / predictions.Rows;
            return Matrix.ApplyElementwiseFunction(predictions, labels,
                (p, l) =>
                norm * (p - l)
            );
        }
    }
}
