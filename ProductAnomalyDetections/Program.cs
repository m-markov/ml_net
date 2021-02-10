using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;

namespace ProductAnomalyDetections
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "TestData", "ProductData.txt");
        //assign the Number of records in dataset file to constant variable
        const int _docsize = 35;

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            IDataView dataView = mlContext.Data.LoadFromTextFile<ProductData>(path: _dataPath, hasHeader: true, separatorChar: ',');

            DetectSpike(mlContext, _docsize, dataView);

            DetectChangepoint(mlContext, _docsize, dataView);
        }

        static void DetectSpike(MLContext mlContext, int docSize, IDataView productSales)
        {
            var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(ProductPrediction.Prediction), inputColumnName: nameof(ProductData.numSales), confidence: 95, pvalueHistoryLength: docSize / 4);
            ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView(mlContext));
            IDataView transformedData = iidSpikeTransform.Transform(productSales);
            var predictions = mlContext.Data.CreateEnumerable<ProductPrediction>(transformedData, reuseRowObject: false);
            Console.WriteLine("Alert\tScore\tP-Value");

            foreach (var p in predictions)
            {
                var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";

                if (p.Prediction[0] == 1)
                {
                    results += " <-- Spike detected";
                }

                Console.WriteLine(results);
            }
            Console.WriteLine("");
        }

        static IDataView CreateEmptyDataView(MLContext mlContext)
        {
            // Create empty DataView.
            IEnumerable<ProductData> enumerableData = new List<ProductData>();
            return mlContext.Data.LoadFromEnumerable(enumerableData);
        }

        static void DetectChangepoint(MLContext mlContext, int docSize, IDataView productSales)
        {
            var iidChangePointEstimator = mlContext.Transforms.DetectIidChangePoint(outputColumnName: nameof(ProductPrediction.Prediction), inputColumnName: nameof(ProductData.numSales), confidence: 95, changeHistoryLength: docSize / 4);
            var iidChangePointTransform = iidChangePointEstimator.Fit(CreateEmptyDataView(mlContext));
            IDataView transformedData = iidChangePointTransform.Transform(productSales);
            var predictions = mlContext.Data.CreateEnumerable<ProductPrediction>(transformedData, reuseRowObject: false);
            Console.WriteLine("Alert\tScore\tP-Value\tMartingale value");

            foreach (var p in predictions)
            {
                var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}\t{p.Prediction[3]:F2}";

                if (p.Prediction[0] == 1)
                {
                    results += " <-- alert is on, predicted changepoint";
                }
                Console.WriteLine(results);
            }
            Console.WriteLine("");
        }
    }
}
