using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Collections.Generic;
using System.Linq;

namespace Utility.UI
{
    public static class ImageWriter
    {
        /// <summary>
        /// Writes a greyscale image from a 2d array of double data.
        /// The data is assumed to be in the range [0..1]
        /// </summary>
        /// <param name="data"></param>
        /// <param name="filename"></param>
        public static void GenerateImage(double[,] data, string filename)
        {
            var width = data.GetLength(0);
            var height = data.GetLength(1);
            
            using (var image = new Image<Rgba32>(width, height))
            {
                for (var x = 0; x < width; x++)
                {
                    for (var y = 0; y < height; y++)
                    {
                        var fd = (float)data[x, y];
                        image[x, y] = new Rgba32(fd, fd, fd);

                    }
                }
                image.Save(filename);
            }
        }
    }
}
