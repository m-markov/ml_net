using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ProductAnomalyDetections
{
    public class ProductData
    {
        [LoadColumn(0)]
        public string Month;

        [LoadColumn(1)]
        public float numSales;
    }
}
