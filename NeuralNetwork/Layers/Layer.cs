using System;
using System.Collections.Generic;
using MatrixLib;

namespace NeuralNetwork.Layers
{
    public abstract class Layer
    {
        public Layer(string name)
        {
            Parameters = new Dictionary<string, Matrix>();
            Gradients = new Dictionary<string, Matrix>();
            Name = name;
        }

        public string Name { get; }

        public Guid Guid { get; } = Guid.NewGuid(); // A unique identifier is needed for the adam optimiser.

        public Matrix? Input { get; set; }

        public Matrix? Output { get; set; }

        public Dictionary<string, Matrix> Parameters { get; }

        /// <summary>
        /// Input grandient used in back propagation
        /// </summary>
        public Matrix? InputGradient { get; set; }

        /// <summary>
        /// List of all parameters gradients calculated during back propagation.
        /// </summary>
        public Dictionary<string, Matrix> Gradients { get; }

        public virtual void Forward(Matrix input)
        {
            Input = input;
        }

        public abstract void Backward(Matrix gradient);
    }
}
