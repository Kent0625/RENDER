"use client";

import React, { useState, useMemo } from "react";
import SliderControl from "../ui/SliderControl";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Scatter } from 'recharts';

export default function RocAucModule() {
  const [separation, setSeparation] = useState(20); // Scale 0-50 (mapped to 0-5)
  const [threshold, setThreshold] = useState(50);   // 0-100 (mapped to 0-1)

  // Generate ROC Data
  const { data, auc, currentPoint, ksStat } = useMemo(() => {
     // Simulation: Two Gaussian distributions (Negatives at 0, Positives at separation/10)
     // We simply generate points for the ROC curve analytically or numerically
     // Analytical ROC for two gaussians (equal variance) is related to separation d'
     
     const d_prime = separation / 10;
     const points = [];
     
     // Generate curve points
     for(let f = 0; f <= 100; f++) {
         const fpr = f / 100;
         // Inverse Normal CDF approximation or simple power law for viz
         // Simple heuristic: TPR = FPR^(1 / (1+d_prime)) - this is convex, but ROC usually concave.
         // Better: TPR = 1 - (1 - FPR)^(d_prime + 1) ? No.
         
         // Let's use a sigmoid-like shape based on separation
         // A common approx for ROC: y = x^a where 0 < a < 1. 
         // Higher separation -> curve pushes to top-left (a gets smaller)
         const power = 1 / (1 + d_prime * 1.5);
         const tpr = Math.pow(fpr, power);
         points.push({ fpr, tpr });
     }
     
     // Calculate AUC (trapezoidal rule, or analytical integral of x^p is 1/(p+1))
     // Integral of x^p from 0 to 1 is 1/(p+1).
     // Wait, y = x^p (convex if p>1). We want concave (p < 1).
     // Area under y=x^p is 1/(p+1).
     const power = 1 / (1 + d_prime * 1.5);
     const calculated_auc = 1 / (power + 1); 
     
     // Current Threshold Logic
     // High threshold -> Low FPR, Low TPR (Bottom Left)
     // Low threshold -> High FPR, High TPR (Top Right)
     // Threshold 1.0 -> FPR 0.
     // Threshold 0.0 -> FPR 1.
     // Let's map threshold (0-1) to FPR (1-0).
     const threshVal = threshold / 100;
     const currentFpr = 1 - threshVal; 
     const currentTpr = Math.pow(currentFpr, power);
     
     // KS Statistic: Max diff between TPR and FPR (here simply max vertical distance from diagonal)
     // Max of (x^p - x).
     // Derivative p*x^(p-1) - 1 = 0 => x = p^(1/(1-p)).
     const x_ks = Math.pow(power, 1/(1-power));
     const y_ks = Math.pow(x_ks, power);
     const ks = y_ks - x_ks;

     return { 
         data: points, 
         auc: calculated_auc,
         currentPoint: { x: currentFpr, y: currentTpr },
         ksStat: ks
     };
  }, [separation, threshold]);

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold mb-2">ROC/AUC & KS Chart</h2>
        <p className="text-gray-600 dark:text-gray-400">
            Receiver Operating Characteristic curve and Area Under Curve.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="md:col-span-1 bg-white dark:bg-gray-800/50 p-6 rounded-xl border border-gray-100 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">Controls</h3>
            <SliderControl label="Class Separation (Difficulty)" value={separation} min={0} max={50} onChange={setSeparation} />
            <p className="text-xs text-gray-500 mb-4">Higher separation = Better model performance</p>
            
            <SliderControl label="Decision Threshold" value={threshold} min={0} max={100} onChange={setThreshold} />
            <p className="text-xs text-gray-500">Adjusting threshold moves along the curve</p>
        </div>

        <div className="md:col-span-2 space-y-6">
            <div className="grid grid-cols-2 gap-4">
                 <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-xl border border-indigo-100 dark:border-indigo-800">
                    <p className="text-xs font-medium text-indigo-600 uppercase">AUC Score</p>
                    <p className="text-3xl font-bold mt-1">{auc.toFixed(3)}</p>
                </div>
                <div className="bg-pink-50 dark:bg-pink-900/20 p-4 rounded-xl border border-pink-100 dark:border-pink-800">
                    <p className="text-xs font-medium text-pink-600 uppercase">KS Statistic</p>
                    <p className="text-3xl font-bold mt-1">{ksStat.toFixed(3)}</p>
                </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 h-96">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                        <defs>
                            <linearGradient id="colorTpr" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
                                <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
                            </linearGradient>
                        </defs>
                        <XAxis dataKey="fpr" type="number" domain={[0, 1]} tickFormatter={(val) => val.toFixed(1)} label={{ value: 'False Positive Rate', position: 'insideBottomRight', offset: -10 }} />
                        <YAxis domain={[0, 1]} tickFormatter={(val) => val.toFixed(1)} label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft' }} />
                        <CartesianGrid strokeDasharray="3 3" />
                        <Tooltip formatter={(value: number) => value.toFixed(3)} />
                        
                        <Area type="monotone" dataKey="tpr" stroke="#8884d8" fillOpacity={1} fill="url(#colorTpr)" />
                        
                        {/* Diagonal Reference */}
                        <ReferenceLine segment={[{x:0, y:0}, {x:1, y:1}]} stroke="gray" strokeDasharray="3 3" />
                        
                        {/* Current Point Marker - Simulated with Reference Lines or a separate Scatter layer (tricky in AreaChart) */}
                        {/* We use ReferenceLines to crosshair the point */}
                        <ReferenceLine x={currentPoint.x} stroke="red" strokeDasharray="3 3" />
                        <ReferenceLine y={currentPoint.y} stroke="red" strokeDasharray="3 3" />
                        
                    </AreaChart>
                </ResponsiveContainer>
            </div>
            <div className="text-center text-sm text-gray-500">
                Current Threshold Point: FPR = {currentPoint.x.toFixed(2)}, TPR = {currentPoint.y.toFixed(2)}
            </div>
        </div>
      </div>
    </div>
  );
}
