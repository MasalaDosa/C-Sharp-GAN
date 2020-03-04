using MatrixLib;

namespace NeuralNetwork.Costs
{
    public abstract class Cost
    {
        public Cost(string name)
        {
            Name = name;
        }

        public string Name { get; set; }

        public abstract Matrix Forward(Matrix predictions, Matrix labels);

        public abstract Matrix Backward(Matrix predictions, Matrix labels);
    }
}
