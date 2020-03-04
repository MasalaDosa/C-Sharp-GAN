using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace MatrixLib.Test
{
    [TestClass]
    public class TestMatrix
    {
        [DataTestMethod()]
        [DataRow(0, 0)]
        [DataRow(0, -1)]
        [DataRow(0, 1)]
        [DataRow(-1, 0)]
        [DataRow(1, 0)]
        [DataRow(1, -1)]
        public void TestInvalidConstruction(int r, int c)
        {
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => new Matrix(r, c));
        }

        [DataTestMethod()]
        [DataRow(1, 1)]
        [DataRow(5, 5)]
        [DataRow(9, 9)]

        public void TestValidConstruction(int r, int c)
        {
            var x = new Matrix(r, c);

            Assert.AreEqual(r, x.Rows);
            Assert.AreEqual(c, x.Columns);
            Assert.AreEqual(r * c, x.Count);
            Assert.AreEqual(r * c, x.Data.Length);
        }

        [TestMethod]
        public void TestDataPresentedInCorrectOrder()
        {
            var x = Matrix.FromData(
                new double[6] {
                    1, 2,
                    3, 4,
                    5, 6
                },
                3, 2);

            Assert.AreEqual(3, x.Rows);
            Assert.AreEqual(2, x.Columns);
            Assert.AreEqual(6, x.Count);
            Assert.AreEqual(1, x[0, 0]);
            Assert.AreEqual(2, x[0, 1]);
            Assert.AreEqual(3, x[1, 0]);
            Assert.AreEqual(4, x[1, 1]);
            Assert.AreEqual(5, x[2, 0]);
            Assert.AreEqual(6, x[2, 1]);
        }

        [DataTestMethod]
        [DataRow(-1, -1)]
        [DataRow(-1, 3)]
        [DataRow(-1, 5)]
        [DataRow(6, -1)]
        [DataRow(6, 5)]
        [DataRow(3, 5)]
        public void TestSettingInvalidRowColumn(int row, int column)
        {
            var x = new Matrix(6, 5);

            Assert.ThrowsException<IndexOutOfRangeException>(() => x[row, column] = 1);
        }

        [DataTestMethod]
        [DataRow(-1, -1)]
        [DataRow(-1, 3)]
        [DataRow(-1, 5)]
        [DataRow(6, -1)]
        [DataRow(6, 5)]
        [DataRow(3, 5)]
        public void TestGettingInvalidRowColumn(int row, int column)
        {
            var x = new Matrix(6, 5);

            Assert.ThrowsException<IndexOutOfRangeException>(() => { var _ = x[row, column]; });
        }

        [DataTestMethod]
        [DataRow(-1)]
        [DataRow(30)]
        public void TestSettingInvalidIndex(int index)
        {
            var x = new Matrix(6, 5);

            Assert.ThrowsException<IndexOutOfRangeException>(() => x[index] = 1);
        }

        [DataTestMethod]
        [DataRow(-1)]
        [DataRow(30)]
        public void TestGettingInvalidIndex(int index)
        {
            var x = new Matrix(6, 5);

            Assert.ThrowsException<IndexOutOfRangeException>(() => { var _ = x[index] = 1; });
        }

        [TestMethod]
        public void TestFullSlice()
        {
            var x = Matrix.FromData(
                new double[10]
                {
                    1, 2,
                    3, 4,
                    5, 6,
                    7, 8,
                    9, 10
                },
                5, 2);

            var slice = x.SliceRows(0, 5);

            var expected = Matrix.FromData(
               new double[10]
               {
                    1, 2,
                    3, 4,
                    5, 6,
                    7, 8,
                    9, 10
               },
               5, 2);

            AssertIsAsExpected(expected, slice);
        }

        [TestMethod]
        public void TestPartialSlice()
        {
            var x = Matrix.FromData(
                new double[10]
                {
                    1, 2,
                    3, 4,
                    5, 6,
                    7, 8,
                    9, 10
                },
                5, 2);

            var slice = x.SliceRows(1, 3);

            var expected = Matrix.FromData(
               new double[6]
               {
                    3, 4,
                    5, 6,
                    7, 8,
               },
               3, 2);

            AssertIsAsExpected(expected, slice);
        }

        [DataTestMethod]
        [DataRow(-1, 1)]
        [DataRow(5, 1)]
        [DataRow(1, 0)]
        [DataRow(1, -1)]
        [DataRow(1, 5)]
        public void TestInvalidSlice(int startRow, int count)
        {
            var x = new Matrix(5, 1);

            Assert.ThrowsException<ArgumentOutOfRangeException>(() => x.SliceRows(startRow, count));
        }

        [TestMethod]
        public void TestCanSliceRowsFull()
        {
            var x = Matrix.FromData(
                new double[10]
                {
                    1, 2,
                    3, 4,
                    5, 6,
                    7, 8,
                    9, 10
                },
                5, 2);

            var result = x.CanSliceRows(0, 5);

            Assert.AreEqual(true, result);
        }

        [TestMethod]
        public void TestCanSliceRowsPartial()
        {
            var x = Matrix.FromData(
                new double[10]
                {
                    1, 2,
                    3, 4,
                    5, 6,
                    7, 8,
                    9, 10
                },
                5, 2);

            var result = x.CanSliceRows(1, 3);

            Assert.AreEqual(true, result);
        }

        [DataTestMethod]
        [DataRow(-1, 1)]
        [DataRow(5, 1)]
        [DataRow(1, 0)]
        [DataRow(1, -1)]
        [DataRow(1, 5)]
        public void TestInvalidCanSliceRows(int startRow, int count)
        {
            var x = new Matrix(5, 1);

            var result = x.CanSliceRows(startRow, count);

            Assert.AreEqual(false, result);
        }

        [TestMethod]
        public void TestMean()
        {
            var x = Matrix.UniformRandomised(-10, 10, 10, 10);

            var mean = x.Average();

            Assert.AreEqual(1, mean.Rows);
            Assert.AreEqual(1, mean.Columns);
            Assert.AreEqual(x.Data.Average(), mean[0]);
        }

        [TestMethod]
        public void TestFilled()
        {
            var x = Matrix.Filled(99, 10, 20);

            Assert.IsFalse(x.Data.Any(d => d != 99));
        }

        [TestMethod]
        public void TestOnes()
        {
            var x = Matrix.Ones(15, 25);

            Assert.IsFalse(x.Data.Any(d => d != 1));
        }

        [TestMethod]
        public void TestZeroes()
        {
            var x = Matrix.Zeroes(5, 10);

            Assert.IsFalse(x.Data.Any(d => d != 0));
        }

        [DataTestMethod]
        [DataRow(-10, 10)]
        [DataRow(-20, 20)]
        public void TestRandom(int min, int max)
        {
            var x = Matrix.UniformRandomised(min, max, 5, 5);

            Assert.IsFalse(x.Data.Any(d => d < min || d > max));
        }

        [DataTestMethod]
        [DataRow(10, 10)]
        [DataRow(10, 9)]
        public void TestRandomMinNotLessThanMax(int min, int max)
        {
            Assert.ThrowsException<ArgumentException>(() => Matrix.UniformRandomised(min, max, 5, 5));
        }

        [TestMethod]
        public void TestFromDataMistmatchedSize()
        {
            Assert.ThrowsException<ArgumentException>(() => Matrix.FromData(new double[1] { 0 }, 2, 2));
        }

        [TestMethod]
        public void TestMatrixMultiply2by2_2by2()
        {
            var x = Matrix.FromData(
                new double[4] {
                    1, 2,
                    3, 4
                },
                2, 2);

            var y = Matrix.FromData(
                new double[4] {
                    5, 6,
                    7, 8
                },
                2, 2);

            var z = x.MatrixMultiply(y);

            Matrix expected = Matrix.FromData(
                new double[]  {
                    19, 22,
                    43, 50
                },
                2, 2);

            AssertIsAsExpected(expected, z);
        }

        [TestMethod]
        public void TestMatrixMultiply3by3_3by1()
        {
            var x = Matrix.FromData(
                new double[9] {
                    0.9, 0.3, 0.4,
                    0.2, 0.8, 0.2,
                    0.1, 0.5, 0.6
                },
                3, 3);


            var y = Matrix.FromData(
                new double[3] {
                    0.9,
                    0.1,
                    0.8
                },
                3, 1);

            var z = x.MatrixMultiply(y);

            var expected = Matrix.FromData(
                new double[3] {
                    1.16,
                    0.42,
                    0.62
                },
                3, 1);

            AssertIsAsExpected(z, expected);
        }

        [TestMethod]
        public void TestMatrixMultiply3by3_3by2()
        {
            var x = Matrix.FromData(
                new double[9] {
                    0.9, 0.3, 0.4,
                    0.2, 0.8, 0.2,
                    0.1, 0.5, 0.6
                },
                3, 3);

            var y = Matrix.FromData(
                new double[6] {
                    0.9, 0.1,
                    0.1, 0.2,
                    0.8, 0.3
                },
                3, 2);

            var z = x.MatrixMultiply(y);

            var expected = Matrix.FromData(
                new double[6] {
                    1.16, 0.27,
                    0.42, 0.24,
                    0.62, 0.29
                },
                3, 2);

            AssertIsAsExpected(z, expected);
        }

        [TestMethod]
        public void TestMatrixMulti0pe3by2_3by3()
        {
            var x = Matrix.FromData(
                new double[6] {
                    0.9, 0.1,
                    0.1, 0.2,
                    0.8, 0.3
                },
                3, 2);

            var y = Matrix.FromData(
                new double[9] {
                    0.9, 0.3, 0.4,
                    0.2, 0.8, 0.2,
                    0.1, 0.5, 0.6
                },
                3, 3);

            Assert.ThrowsException<ArgumentException>(() => x.MatrixMultiply(y));
        }

        [TestMethod]
        public void TestTranspose1()
        {
            var x = Matrix.FromData(
                new double[6] {
                    1, 2, 3,
                    4, 5 ,6
                },
                2, 3);

            var z = x.Transpose();

            var expected = Matrix.FromData(
                new double[6] {
                    1, 4,
                    2, 5,
                    3, 6
                },
                3, 2);

            AssertIsAsExpected(expected, z);
        }

        [TestMethod]
        public void TestTranspose2()
        {
            var x = Matrix.FromData(
                new double[24] {
                    1, 2, 3, 4, 5, 6,
                    7, 8 ,9, 10, 11, 12,
                    13, 14 ,15, 16, 17, 18,
                    19, 20, 21, 22, 23, 24
                },
                4, 6);


            var z = x.Transpose();

            var expected = Matrix.FromData(
                new double[24] {
                    1, 7, 13, 19,
                    2, 8, 14, 20,
                    3, 9, 15, 21,
                    4, 10, 16, 22,
                    5, 11, 17, 23,
                    6, 12, 18, 24
                },
                6, 4);

            AssertIsAsExpected(expected, z);
        }

        [TestMethod]
        public void TestTranspose3()
        {
            var x = Matrix.FromData(
                new double[6] {
                    1, 2, 3, 4, 5, 6
                },
                1, 6);

            Matrix z = x.Transpose();

            var expected = Matrix.FromData(
                new double[6] {
                    1,
                    2,
                    3,
                    4,
                    5,
                    6
                },
                6, 1);

            AssertIsAsExpected(expected, z);
        }

        [TestMethod]
        public void TestTranspose4()
        {
            var x = Matrix.FromData(
                new double[6] {
                    1,
                    2,
                    3,
                    4,
                    5,
                    6
                },
                6, 1);

            Matrix z = x.Transpose();

            var expected = Matrix.FromData(
                 new double[6] {
                    1, 2, 3, 4, 5, 6
                 },
                 1, 6);

            AssertIsAsExpected(expected, z);
        }

        [TestMethod]
        public void TestApplyFunction3by2times5()
        {

            var x = Matrix.FromData(
                new double[6] {
                    0.9d, 0.1d,
                    0.1d, 0.2d,
                    0.8d, 0.3d},
                    3, 2);

            Matrix z = Matrix.ApplyElementwiseFunction(x, d => d * 5);

            var expected = Matrix.FromData(
               new double[6] {
                    4.5, 0.5,
                    0.5, 1.0,
                    4.8, 1.5 },
                   3, 2);

            AssertIsAsExpected(expected, z);
        }

        [TestMethod]
        public void TestApplyFunction3by2Times3by2()
        {

            var x = Matrix.FromData(
                new double[6] {
                    0.9, 0.1,
                    0.1, 0.2,
                    0.8, 0.3 },
                3, 2);

            var y = Matrix.FromData(
                new double[6] {
                    2, 1,
                    1, 2,
                    2, 1 },
                3, 2);

            Matrix z = Matrix.ApplyElementwiseFunction(x, y, (a, b) => a * b);

            var expected = Matrix.FromData(
               new double[6] {
                    1.8, 0.1,
                    0.1, 0.4,
                    1.6, 0.3 },
                   3, 2);

            AssertIsAsExpected(expected, z);
        }

        [TestMethod]
        public void TestApplyFunction3by2Times2by3AllowedAsSameNuberOfElements()
        {

            var x = Matrix.FromData(
                new double[6] {
                    0.9, 0.1,
                    0.1, 0.2,
                    0.8, 0.3 },
                3, 2);

            var y = Matrix.FromData(
                new double[6] {
                    2, 1, 1,
                    2, 2, 1 },
                2, 3);

            var z = Matrix.ApplyElementwiseFunction(x, y, (a, b) => a * b);

            var expected = Matrix.FromData(
              new double[6] {
                    1.8, 0.1,
                    0.1, 0.4,
                    1.6, 0.3 },
                  3, 2);

            AssertIsAsExpected(expected, z);
        }

        [TestMethod]
        public void TestApplyFunction3by2Times2by2Mismatch()
        {

            var x = Matrix.FromData(
                new double[6] {
                    0.9, 0.1,
                    0.1, 0.2,
                    0.8, 0.3 },
                3, 2);

            var y = Matrix.FromData(
                new double[4] {
                    2, 1,
                    1, 2 },
                2, 2);

            Assert.ThrowsException<ArgumentException>(() => Matrix.ApplyElementwiseFunction(x, y, (a, b) => a * b));
        }

        [TestMethod]
        public void TestApplyFunction2by2Plus2by2Plus2by2()
        {

            var x = Matrix.FromData(
                new double[4] {
                    1, 2,
                    3, 4 },
                2, 2);

            var y = Matrix.FromData(
                new double[4] {
                    5, 6,
                    7, 8},
                2, 2);

            var z = Matrix.FromData(
                new double[4] {
                    9, 10,
                    11, 12 },
                2, 2);

            var w = Matrix.ApplyElementwiseFunction(x, y, z, (a, b, c) => a + b + c);

            var expected = Matrix.FromData(
              new double[4] {
                    15, 18,
                    21, 24 },
                  2, 2);

            AssertIsAsExpected(expected, z);
        }

        [TestMethod]
        public void TestApplyFunction2by2Plus3by3Plus2by2Mismatch()
        {

            var x = Matrix.FromData(
                new double[4] {
                    1, 2,
                    3, 4 },
                2, 2);

            var y = Matrix.FromData(
                new double[9] {
                    5, 6, 7,
                    8, 9, 10,
                    11, 12, 13},
                3, 3);

            var z = Matrix.FromData(
                new double[4] {
                    9, 10,
                    11, 12},
                2, 2);

            Assert.ThrowsException<ArgumentException>(() => Matrix.ApplyElementwiseFunction(x, y, z, (a, b, c) => a + b + c));
        }

        [TestMethod]
        public void TestApplyFunction2by2Plus2by2Plus3by3Mismatch()
        {

            var x = Matrix.FromData(
                new double[4] {
                    1, 2,
                    3, 4},
                2, 2);

            var y = Matrix.FromData(
                new double[4] {
                    5, 6,
                    7, 8},
                2, 2);

            var z = Matrix.FromData(
                new double[9] {
                    9, 10, 11,
                    12, 13, 14,
                    15, 16, 17},
                3, 3);

            Assert.ThrowsException<ArgumentException>(() => Matrix.ApplyElementwiseFunction(x, y, z, (a, b, c) => a + b + c));
        }

        static void AssertIsAsExpected(Matrix expected, Matrix actual)
        {
            Assert.AreEqual(expected.Rows, actual.Rows);
            Assert.AreEqual(expected.Columns, actual.Columns);
            for (int i = 0; i < actual.Count; i++)
            {
                Assert.IsTrue(actual[i] - expected[i] < 0.001, $"{actual.ToString()}");
            }
        }
    }
}
