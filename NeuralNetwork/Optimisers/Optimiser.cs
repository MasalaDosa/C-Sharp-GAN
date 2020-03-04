using NeuralNetwork.Layers;

namespace NeuralNetwork.Optimisers
{
    public abstract class Optimiser
    {
        public Optimiser(string name, double learningRate)
        {
            Name = name;
            LearningRate = learningRate;
        }

        public string Name { get; set; }

        public double LearningRate { get; protected set; }

        public abstract void Update(Layer layer);
    }
}
