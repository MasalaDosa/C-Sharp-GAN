using MatrixLib;

namespace NeuralNetwork.Layers.Activations
{
    public class LeakyReLUActivation : Activation
    {
        private double alpha;

        public LeakyReLUActivation(string name) : base(name)
        {
            alpha = 0d;
        }

        public LeakyReLUActivation(double alpha, string name = "relu") : base($"{name} ({alpha})")
        {
            this.alpha = alpha;
        }

        public override void Forward(Matrix input)
        {
            base.Forward(input);
            Output = Matrix.ApplyElementwiseFunction(input,
                i =>
                i > 0 ? i : i * alpha
            );
        }

        public override void Backward(Matrix gradient)
        {
            InputGradient = Matrix.ApplyElementwiseFunction(gradient, Output!,
                (g, o) =>
                g * (o > 0 ? 1 : alpha)
            );
        }
    }
}
