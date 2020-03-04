using System;
using MatrixLib;

namespace NeuralNetwork.Layers.Activations
{
    public class SigmoidActivation : Activation
    {
        public SigmoidActivation(string name = "sigmoid") : base(name) { }

        public override void Forward(Matrix input)
        {
            base.Forward(input);
            var exp = Matrix.ApplyElementwiseFunction(input,
                i => Math.Exp(i)
            );

            Output = Matrix.ApplyElementwiseFunction(exp,
                ex =>
                ex / (1 + ex)
            );
        }

        public override void Backward(Matrix gradient)
        {
            InputGradient = Matrix.ApplyElementwiseFunction(gradient, Output!,
                (g, o) =>
                g * o * (1 - o)
            );
        }
    }
}
