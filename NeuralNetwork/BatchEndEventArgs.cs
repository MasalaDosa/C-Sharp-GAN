namespace NeuralNetwork
{
    public class BatchEndEventArgs
    {
        public BatchEndEventArgs(
            int epoch,
            int batch,
            double loss)
        {
            Epoch = epoch;
            Batch = batch;
            Loss = loss;
        }

        public int Epoch { get; }

        public int Batch { get; }

        public double Loss { get; }

        public override string ToString() => $"Epoch {Epoch} Batch {Batch} Loss {Loss}.";
    }
}
