// This file was auto-generated by ML.NET Model Builder. 

using System;
using StockPricePredictionML.Model;

namespace StockPricePredictionML.ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create single instance of sample data from first line of dataset for model input
            ModelInput sampleData = new ModelInput()
            {
                Date = @"2016-01-05",
                Symbol = @"WLTW",
                Open = 123.43F,
                Low = 122.31F,
                High = 126.25F,
                Volume = 2163600F,
            };

            // Make a single prediction on the sample data and print results
            var predictionResult = ConsumeModel.Predict(sampleData);

            Console.WriteLine("Using model to make single prediction -- Comparing actual Close with predicted Close from sample data...\n\n");
            Console.WriteLine($"Date: {sampleData.Date}");
            Console.WriteLine($"Symbol: {sampleData.Symbol}");
            Console.WriteLine($"Open: {sampleData.Open}");
            Console.WriteLine($"Low: {sampleData.Low}");
            Console.WriteLine($"High: {sampleData.High}");
            Console.WriteLine($"Volume: {sampleData.Volume}");
            Console.WriteLine($"\n\nPredicted Close: {predictionResult.Score}\n\n");
            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }
    }
}
