"use client";

import React, { useState, useMemo } from "react";
import SliderControl from "../ui/SliderControl";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

export default function CorrelationModule() {
  const [targetR, setTargetR] = useState(85); // -100 to 100
  
  // Generate Data
  const { data, metrics } = useMemo(() => {
     const r = targetR / 100;
     const n = 100;
     const points = [];
     
     // Generate correlated data using Box-Muller or simple mixing
     // y = r*x + sqrt(1-r^2)*noise
     // Standard normal variables
     
     let meanX = 0; 
     let meanY = 0;
     let sumX = 0;
     let sumY = 0;
     let sumXY = 0;
     let sumX2 = 0;
     let sumY2 = 0;

     for(let i=0; i<n; i++) {
         const u1 = Math.random();
         const u2 = Math.random();
         const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2); // Norm(0,1)
         
         const u3 = Math.random();
         const u4 = Math.random();
         const z2 = Math.sqrt(-2.0 * Math.log(u3)) * Math.cos(2.0 * Math.PI * u4); // Norm(0,1)
         
         const x = z1;
         // Correlate y with x
         const y = r * z1 + Math.sqrt(1 - r*r) * z2;
         
         points.push({ x, y });
         
         sumX += x;
         sumY += y;
         sumXY += x*y;
         sumX2 += x*x;
         sumY2 += y*y;
     }
     
     // Calculate Pearson R from sample to confirm
     const num = n * sumXY - sumX * sumY;
     const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
     const actualR = den === 0 ? 0 : num / den;
     const r2 = actualR * actualR;
     
     return { data: points, metrics: { r: actualR, r2 } };
  }, [targetR]);

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold mb-2">Correlation: Pearson's R & R²</h2>
        <p className="text-gray-600 dark:text-gray-400">
            Measure the strength and direction of linear relationships.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="md:col-span-1 bg-white dark:bg-gray-800/50 p-6 rounded-xl border border-gray-100 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">Controls</h3>
            <SliderControl label="Target Correlation (R)" value={targetR} min={-99} max={99} onChange={setTargetR} step={5} />
            <div className="mt-2 text-xs text-gray-500 flex justify-between">
                <span>-1.0 (Neg)</span>
                <span>0.0 (None)</span>
                <span>+1.0 (Pos)</span>
            </div>

            <div className="mt-8 space-y-4">
                 <div className="p-4 bg-blue-50 dark:bg-blue-900/10 rounded-lg border border-blue-100 dark:border-blue-800">
                     <p className="text-xs text-gray-500 uppercase">Pearson's R (Sample)</p>
                     <p className="text-2xl font-bold text-blue-700 dark:text-blue-400">{metrics.r.toFixed(3)}</p>
                 </div>
                 <div className="p-4 bg-purple-50 dark:bg-purple-900/10 rounded-lg border border-purple-100 dark:border-purple-800">
                     <p className="text-xs text-gray-500 uppercase">R-Squared (R²)</p>
                     <p className="text-2xl font-bold text-purple-700 dark:text-purple-400">{metrics.r2.toFixed(3)}</p>
                 </div>
            </div>
        </div>

        <div className="md:col-span-2 space-y-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 h-96">
                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="x" type="number" name="X" label={{ value: 'Variable X', position: 'insideBottom', offset: -10 }} />
                        <YAxis dataKey="y" type="number" name="Y" label={{ value: 'Variable Y', angle: -90, position: 'insideLeft' }} />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Scatter name="Data" data={data} fill="#8884d8" />
                        
                        {/* Quadrant Lines */}
                        <ReferenceLine x={0} stroke="#ccc" />
                        <ReferenceLine y={0} stroke="#ccc" />
                    </ScatterChart>
                </ResponsiveContainer>
            </div>
            <div className="text-sm text-gray-500 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <strong>Insight:</strong> R² represents the proportion of variance in Y explained by X. If R=0.8, R²=0.64, meaning 64% of the variation is explained.
            </div>
        </div>
      </div>
    </div>
  );
}
