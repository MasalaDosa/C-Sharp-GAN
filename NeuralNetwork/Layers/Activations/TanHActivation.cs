using System;
using MatrixLib;

namespace NeuralNetwork.Layers.Activations
{
    public class TanhActivation : Activation
    {
        public TanhActivation(string name = "tanh") : base(name) { }

        public override void Forward(Matrix input)
        {
            base.Forward(input);
            Output = Matrix.ApplyElementwiseFunction(input,
                i =>
                Math.Tanh(i)
            );
        }

        public override void Backward(Matrix gradient)
        {
            InputGradient = Matrix.ApplyElementwiseFunction(gradient, Output!,
                (g, o) =>
                g * (1 - Math.Pow(o, 2))
            );
        }
    }
}
