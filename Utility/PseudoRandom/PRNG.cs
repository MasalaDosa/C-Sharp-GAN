using System;

namespace Utility.PseudoRandom
{
    /// <summary>
    /// Base class for Random number generators
    /// </summary>
    public abstract class PRNG
    {
        /// <summary>
        /// Get a default instance of the Basic PRNG
        /// </summary>
        public static PRNG Basic { get; } = new BasicPRNG();

        /// <summary>
        /// Returns a uniformly distributed double from 0. to < 1.
        /// </summary>
        /// <returns></returns>
        public abstract double GetUniformDouble();

        /// <summary>
        /// Returns a uniformly distributed double from min to < max
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public abstract double GetUniformDouble(double min, double max);

        /// <summary>
        /// Uniformly distributed int32
        /// </summary>
        /// <returns></returns>
        public abstract int GetUniformInt32();

        /// <summary>
        /// Uniformly distrubted int32 from min to < max
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public abstract int GetUniformInt32(int min, int max);

        /// <summary>
        /// Get normal (Gaussian) random sample with mean 0 and standard deviation 1
        /// </summary>
        /// <returns></returns>
        public virtual double GetNormalDouble()
        {
            // Use Box-Muller algorithm
            double u1 = GetUniformDouble();
            double u2 = GetUniformDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            return r * Math.Sin(theta);
        }

        /// <summary>
        /// Get normal (Gaussian) random sample with specified mean and standard deviation
        /// </summary>
        /// <param name="mean"></param>
        /// <param name="standardDeviation"></param>
        /// <returns></returns>
        public virtual double GetNormalDouble(double mean, double standardDeviation)
        {
            if (standardDeviation <= 0.0)
            {
                throw new ArgumentOutOfRangeException($"{nameof(standardDeviation)} must be positive.");
            }
            return mean + standardDeviation * GetNormalDouble();
        }
    }
}
