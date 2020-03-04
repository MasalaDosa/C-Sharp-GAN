using System;
using System.Linq;
using System.Text;
using Utility.PseudoRandom;

namespace MatrixLib
{
    /// <summary>
    /// Basic matrix implementation.
    /// </summary>
    public class Matrix
    {
        public Matrix(int rows, int columns)
        {
            Rows = rows > 0 ? rows : throw new ArgumentOutOfRangeException($"{nameof(rows)} must be +ve.");
            Columns = columns > 0 ? columns : throw new ArgumentOutOfRangeException($"{nameof(columns)} must be +ve.");
            Count = rows * columns;
            Data = new double[Count];
        }

        public int Rows { get; }

        public int Columns { get; }

        public int Count { get; }

        public double[] Data { get; }

        public double this[int index]
        {
            get
            {
                return Data[index];
            }
            set
            {
                Data[index] = value;
            }
        }

        public double this[int row, int column]
        {
            get
            {
                return Data[FindIndexFromRowAndColumn(row, column)];
            }
            set
            {

                Data[FindIndexFromRowAndColumn(row, column)] = value;
            }
        }

        private long FindIndexFromRowAndColumn(int row, int column)
        {
            if (row < 0 || row >= Rows) throw new IndexOutOfRangeException($"{nameof(row)}");
            if (column < 0 || column >= Columns) throw new IndexOutOfRangeException($"{nameof(column)}");
            return row * Columns + column;
        }

        public Matrix SliceRows(int startRow, int countOfRows)
        {
            if (countOfRows < 1)
            {
                throw new ArgumentOutOfRangeException($"{nameof(countOfRows)}.");
            }
            if (startRow < 0  || startRow >= Rows)
            {
                throw new ArgumentOutOfRangeException($"{nameof(startRow)}.");
            }
            if (startRow + countOfRows > Rows)
            {
                throw new ArgumentOutOfRangeException($"{nameof(countOfRows)}.");
            }
            var startIdx = Columns * startRow;
            var countOfElements = Columns * countOfRows;

            var slicedData = Data.Skip(startIdx).Take(countOfElements).ToArray();
            return FromData(slicedData, slicedData.Length / Columns, Columns);
        }

        public bool CanSliceRows(int startRow, int countOfRows)
        {
            var startIdx = Columns * startRow;
            var countOfElements = Columns * countOfRows;
            if (startIdx < 0 || countOfElements <= 0 || startIdx + countOfElements > Data.Length)
            {
                return false;
            }

            return true;
        }

        public Matrix Average()
        {
            return Filled(Data.Average(), 1, 1);
        }

        public Matrix MatrixMultiply(Matrix other)
        {
            if (Columns != other.Rows)
            {
                throw new ArgumentException($"Cols must be equal to {nameof(other)} Rows.");
            }

            Matrix result = new Matrix(Rows, other.Columns);
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < other.Columns; j++)
                {
                    result[i, j] = 0;
                    for (int k = 0; k < Columns; k++)
                    {
                        result[i, j] += this[i, k] * other[k, j];
                    }
                }
            }
            return result;
        }

        public Matrix Transpose()
        {
            Matrix result = new Matrix(Columns, Rows);
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    result[j, i] = this[i, j];
                }
            }
            return result;
        }

        public void ShuffleRows()
        {
            for (int i = Rows - 1; i > 0; i--)
            {
                // Swap element "i" with a random earlier element it (or itself)
                var swapIndex = PRNG.Basic.GetUniformInt32(0, i + 1);
                if (swapIndex != i)
                {
                    SwapRows(i, swapIndex);
                }
              
            }
        }

        private void SwapRows(int r1, int r2)
        {
            for (int i = 0; i < Columns; i++)
            {

                var r1iIndex = FindIndexFromRowAndColumn(r1, i);
                var r2iIndex = FindIndexFromRowAndColumn(r2, i);

                var tmp = Data[r1iIndex];
                Data[r1iIndex] = Data[r2iIndex];
                Data[r2iIndex] = tmp;
            }
        }

        public override string ToString()
        {
            const int MAX_ELEMENTS_TO_SHOW = 5;
            int elememtsToShow = Math.Min(MAX_ELEMENTS_TO_SHOW, Count);
            StringBuilder toString = new StringBuilder();
            toString.Append($"{Rows} by {Columns}");
            toString.Append(" : ");
            toString.Append(Data.Take(elememtsToShow).Select(d => d.ToString()).Aggregate((a, b) => $"{a}, {b}"));
            return toString.ToString();
        }

        public static Matrix Filled(double fillWith, int rows, int columns)
        {
            var result = new Matrix(rows, columns);
            for (int i = 0; i < result.Count; i++)
            {
                result[i] = fillWith;
            }
            return result;
        }

        public static Matrix Zeroes(int rows, int columns)
        {
            return Filled(0d, rows, columns);
        }

        public static Matrix Ones(int rows, int columns)
        {
            return Filled(1d, rows, columns);
        }

        public static Matrix UniformRandomised(double min, double max, int rows, int columns)
        {
            if (min >= max) throw new ArgumentException($"{nameof(min)} must be less than {nameof(max)}.");
            var result = new Matrix(rows, columns);
            for (int i = 0; i < result.Count; i++)
            {
                result[i] = PRNG.Basic.GetUniformDouble(min, max);
            }
            return result;
        }

        public static Matrix NormalRandomised(double mean, double standardDeviation, int rows, int columns)
        {
            var result = new Matrix(rows, columns);
            for (int i = 0; i < result.Count; i++)
            {
                result[i] = PRNG.Basic.GetNormalDouble(mean, standardDeviation);
            }
            return result;
        }

        public static Matrix FromData(double[] data, int rows, int columns)
        {
            var result = new Matrix(rows, columns);
            if (data.Length != result.Count)
            {
                throw new ArgumentException($"{nameof(data)} does not match {rows} and {columns}.");
            }
            //result.Data = data;

            for (int i = 0; i < result.Count; i++)
            {
                result[i] = data[i];
            }
            return result;
        }

        public static Matrix ApplyElementwiseFunction(Matrix first, Func<double, double> f)
        {
            Matrix result = new Matrix(first.Rows, first.Columns);

            for (var i = 0; i < first.Count; i++)
            {
                result.Data[i] = f(first[i]);
            }

            return result;
        }

        public static Matrix ApplyElementwiseFunction(Matrix first, Matrix second, Func<double, double, double> f)
        {
            if (first.Count != second.Count)
            {
                throw new ArgumentException($"{nameof(first)} must have the name number of elements as {nameof(second)}.");
            }

            Matrix result = new Matrix(first.Rows, first.Columns);

            for (var i = 0; i < first.Count; i++)
            {
                result.Data[i] = f(first[i], second[i]);
            }

            return result;
        }

        public static Matrix ApplyElementwiseFunction(Matrix first, Matrix second, Matrix third, Func<double, double, double, double> f)
        {
            if (first.Count != second.Count || second.Count != third.Count)
            {
                throw new ArgumentException($"{nameof(first)} must have the name number of elements as {nameof(second)} and {nameof(third)}.");
            }

            Matrix result = new Matrix(first.Rows, first.Columns);

            for (var i = 0; i < first.Count; i++)
            {
                result.Data[i] = f(first[i], second[i], third[i]);
            }

            return result;
        }
    }
}
