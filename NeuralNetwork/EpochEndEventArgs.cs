namespace NeuralNetwork
{
    public class EpochEndEventArgs
    {
        public EpochEndEventArgs(
            int epoch,
            double loss)
        {
            Epoch = epoch;
            Loss = loss;
        }

        public int Epoch { get; }

        public double Loss { get; }

        public override string ToString() => $"Epoch {Epoch} Loss {Loss}";
    }
}
