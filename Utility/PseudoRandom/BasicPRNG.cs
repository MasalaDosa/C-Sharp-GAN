using System;

namespace Utility.PseudoRandom
{
    /// <summary>
    /// A Random Number generator that uses the basic System.Random internally.
    /// </summary>
    public class BasicPRNG : PRNG
    {
        private readonly Random rng;

        public BasicPRNG()
        {
            rng = new Random();
        }

        public BasicPRNG(int seed)
        {
            rng = new Random(seed);
        }

        public override double GetUniformDouble()
        {
            return rng.NextDouble();
        }

        public override double GetUniformDouble(double min, double max)
        {
            return min + GetUniformDouble() * (max - min);
        }

        public override int GetUniformInt32()
        {
            return rng.Next();
        }

        public override int GetUniformInt32(int min, int max)
        {
            return rng.Next(min, max);
        }
    }
}
