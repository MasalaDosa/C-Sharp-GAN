using System;
using System.Collections.Generic;
using System.Linq;

namespace Utility.UI
{
    /// <summary>
	/// Contains various utility functions for a console based UI.
	/// </summary>
    public static class ConsoleUI
    {
        private static string DateTimeString => $"{DateTime.Now.ToString("f")} : ";

        /// <summary>
        /// Writes a blank line prepended with the current local date time.
        /// </summary>
        public static void WriteLine()
        {
            Console.WriteLine($"{DateTimeString}");
        }

        /// <summary>
        /// Writes a string prepended with the current local date time.
        /// </summary>
        /// <param name="s"></param>
        public static void WriteLine(string s)
        {
            Console.WriteLine($"{DateTimeString}{s}");
        }

        /// <summary>
        /// Displays a menu consisting of a title and anumber of options.
        /// Waits for an option to be selected and then returns the selected index (1 based).
        /// </summary>
        /// <param name="title"></param>
        /// <param name="options"></param>
        /// <returns></returns>
        public static int ShowMenu(string title, params string[] options)
        {
            if (options.Length == 0)
            {
                throw new ArgumentException($"{nameof(options)} should have at least one option.");
            }
            WriteLine(title);
            WriteLine(new string('-', title.Length));
            for (int i = 0; i < options.Length; i++)
            {
                WriteLine($"{i + 1}. {options[i]}");
            }
            return PromptForNumber(options.Length);
        }

        private static int PromptForNumber(int max)
        {
            int answer;
            do
            {
                WriteLine($"Please enter a number from 1 to {max}.");
                if (int.TryParse(Console.ReadLine(), out answer) &&
                    answer >= 1 &&
                    answer <= max)
                {
                    return answer;
                }
            } while (true);   
        }

        /// <summary>
        /// Prompts for a comma separated list of integers within a specific range.
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public static IEnumerable<int> PromptForListOfIntegers(int min, int max)
        {
            WriteLine($"Please enter a comma separated list of integers from {min} to {max} inclusive:");
            var line = Console.ReadLine();
            var items = line.Split(',', StringSplitOptions.RemoveEmptyEntries);
            foreach (var item in items)
            {
                int answer;
                if (int.TryParse(item, out answer) &&
                   answer >= min &&
                   answer <= max)
                {
                    yield return answer;
                }
                else
                {
                    WriteLine($"Rejected '{item}'.");
                }
            }
            yield break;
        }


        /// <summary>
        /// Displays a representation of the specified double data(e.g an MNIST char).
        /// </summary>
        /// <param name="inputs">The data to display arranged as rows by columns.</param>
        /// <param name="min">The minimum value of the data (which will be inferred from the data if omitted).</param>
        /// <param name="max">The maximum value of the data (which will be inferred from the data if omitted).</param>
        /// <param name="width">The width of the data (it is the callers responsibility to ensure this fits on the screen.</param>
        /// <param name="height">The height of the data (it is the callers responsibility to ensure this fits on the screen.</param>
        public static void DisplayDataAsText(double[] inputs, double? min, double? max, int width = 28, int height = 28)
        {
            if (inputs.Length != width * height)
            {
                throw new ArgumentException($"{nameof(inputs)} should have exactly {width * height} elements.");
            }
            min = min ?? inputs.Min();
            max = max ?? inputs.Max();
            var scale = max - min;
            WriteLine();
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var d = (inputs[y * 28 + x] - min) / scale;
                    if (d >= 0.9) Console.Write("*");
                    else if (d >= 0.8) Console.Write("#");
                    else if (d >= 0.7) Console.Write("%");
                    else if (d >= 0.6) Console.Write("+");
                    else if (d >= 0.5) Console.Write("-");
                    else if (d >= 0.4) Console.Write(":");
                    else if (d >= 0.3) Console.Write("'");
                    else if (d >= 0.2) Console.Write("~");
                    else if (d >= 0.1) Console.Write(".");
                    else Console.Write(" ");
                }
                Console.Write(Environment.NewLine);
            }
        }
    }
}
