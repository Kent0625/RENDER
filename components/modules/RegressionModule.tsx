"use client";

import React, { useState, useMemo } from "react";
import SliderControl from "../ui/SliderControl";
import { ComposedChart, Scatter, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function RegressionModule() {
  const [noise, setNoise] = useState(10); // 0-50
  
  // Generate Data
  const { data, metrics } = useMemo(() => {
     const points = [];
     const n = 50;
     let sumSqErr = 0;
     let sumAbsErr = 0;
     
     // True line: y = 2x + 10
     for(let i=0; i<n; i++) {
         const x = i * 2;
         const trueY = 2 * x + 10;
         // Random noise: Box-Muller or just Math.random approx
         // (Math.random() - 0.5) * 2 * noise
         const err = (Math.random() - 0.5) * 2 * noise * 3; 
         const y = trueY + err;
         
         // Prediction is the true line (simplification for teaching "Error")
         // Or we could fit a line, but measuring error against the "Perfect Model" is also instructive
         // Let's assume we fit a line.
         // Simple Linear Regression:
         // meanX, meanY...
         // Let's just use the true line as the "Prediction" to show how noise affects metrics 
         // against the ground truth trend.
         const predY = trueY; 
         
         const residual = y - predY;
         sumSqErr += residual * residual;
         sumAbsErr += Math.abs(residual);
         
         points.push({ x, y, predY });
     }
     
     const mse = sumSqErr / n;
     const rmse = Math.sqrt(mse);
     const mae = sumAbsErr / n;
     
     return { data: points, metrics: { mse, rmse, mae } };
  }, [noise]);

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold mb-2">Regression Metrics</h2>
        <p className="text-gray-600 dark:text-gray-400">
            Visualize how data noise impacts error metrics (RMSE, MSE, MAE).
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="md:col-span-1 bg-white dark:bg-gray-800/50 p-6 rounded-xl border border-gray-100 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">Controls</h3>
            <SliderControl label="Noise Level (Error)" value={noise} min={0} max={50} onChange={setNoise} />
            <div className="mt-8 space-y-4">
                 <div className="p-4 bg-red-50 dark:bg-red-900/10 rounded-lg border border-red-100 dark:border-red-800">
                     <p className="text-xs text-gray-500 uppercase">RMSE</p>
                     <p className="text-2xl font-bold text-red-700 dark:text-red-400">{metrics.rmse.toFixed(2)}</p>
                 </div>
                 <div className="p-4 bg-orange-50 dark:bg-orange-900/10 rounded-lg border border-orange-100 dark:border-orange-800">
                     <p className="text-xs text-gray-500 uppercase">MSE</p>
                     <p className="text-2xl font-bold text-orange-700 dark:text-orange-400">{metrics.mse.toFixed(2)}</p>
                 </div>
                 <div className="p-4 bg-yellow-50 dark:bg-yellow-900/10 rounded-lg border border-yellow-100 dark:border-yellow-800">
                     <p className="text-xs text-gray-500 uppercase">MAE</p>
                     <p className="text-2xl font-bold text-yellow-700 dark:text-yellow-400">{metrics.mae.toFixed(2)}</p>
                 </div>
            </div>
        </div>

        <div className="md:col-span-2 space-y-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 h-96">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={data} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="x" type="number" label={{ value: 'X', position: 'insideBottomRight', offset: -10 }} />
                        <YAxis label={{ value: 'Y', angle: -90, position: 'insideLeft' }} />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Legend />
                        <Scatter name="Actual Data" dataKey="y" fill="#8884d8" />
                        <Line type="monotone" dataKey="predY" stroke="#ff7300" dot={false} strokeWidth={2} name="Trend Line" />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
            <div className="text-sm text-gray-500 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <strong>Note:</strong> RMSE penalizes large errors more heavily than MAE due to squaring. Notice how RMSE grows faster than MAE as noise increases.
            </div>
        </div>
      </div>
    </div>
  );
}
