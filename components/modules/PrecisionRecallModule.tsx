"use client";

import React, { useState, useMemo } from "react";
import SliderControl from "../ui/SliderControl";
import ConfusionMatrixViz from "../ui/ConfusionMatrixViz";
import { generateMultiClassMatrix } from "@/lib/utils";

export default function PrecisionRecallModule() {
  const [mode, setMode] = useState<"binary" | "multi">("binary");
  
  const [tp, setTp] = useState(60);
  const [tn, setTn] = useState(80);
  const [fp, setFp] = useState(20);
  const [fn, setFn] = useState(40);

  const [samples, setSamples] = useState(300);
  const [noise, setNoise] = useState(20);

  // Reusing logic (would be better to extract hooks, but inline is fine for speed)
  const binaryMatrix = useMemo(() => [[tn, fp], [fn, tp]], [tp, tn, fp, fn]);
  
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;

  const multiMatrix = useMemo(() => {
    return generateMultiClassMatrix(samples, noise);
  }, [samples, noise]);

  const multiMetrics = useMemo(() => {
      // Calculate per class Precision/Recall
      const metrics = [];
      for(let i=0; i<3; i++) {
          const tp = multiMatrix[i][i];
          const fp = multiMatrix.reduce((acc, row) => acc + row[i], 0) - tp;
          const fn = multiMatrix[i].reduce((acc, val) => acc + val, 0) - tp;
          
          metrics.push({
              class: ["Cat", "Dog", "Bird"][i],
              precision: tp / (tp + fp) || 0,
              recall: tp / (tp + fn) || 0
          });
      }
      return metrics;
  }, [multiMatrix]);

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold mb-2">Precision & Recall</h2>
        <div className="grid grid-cols-2 gap-4 text-sm text-gray-600 dark:text-gray-400">
            <p><strong>Precision:</strong> How many retrieved items are relevant? (TP / TP+FP)</p>
            <p><strong>Recall:</strong> How many relevant items are retrieved? (TP / TP+FN)</p>
        </div>
      </div>

      <div className="flex space-x-4 bg-gray-100 dark:bg-gray-800 p-1 rounded-lg w-fit">
        <button onClick={() => setMode("binary")} className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${mode === 'binary' ? 'bg-white shadow text-blue-600' : 'text-gray-500'}`}>Binary</button>
        <button onClick={() => setMode("multi")} className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${mode === 'multi' ? 'bg-white shadow text-blue-600' : 'text-gray-500'}`}>Multi-class</button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="md:col-span-1 bg-white dark:bg-gray-800/50 p-6 rounded-xl border border-gray-100 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">Controls</h3>
            {mode === "binary" ? (
                <>
                    <SliderControl label="True Positives (TP)" value={tp} min={0} max={100} onChange={setTp} />
                    <SliderControl label="False Positives (FP)" value={fp} min={0} max={100} onChange={setFp} />
                    <SliderControl label="False Negatives (FN)" value={fn} min={0} max={100} onChange={setFn} />
                    <div className="opacity-50 pointer-events-none">
                        <SliderControl label="True Negatives (TN) - N/A" value={tn} min={0} max={100} onChange={setTn} />
                    </div>
                </>
            ) : (
                <>
                    <SliderControl label="Total Samples" value={samples} min={30} max={600} step={30} onChange={setSamples} />
                    <SliderControl label="Noise Level (%)" value={noise} min={0} max={90} onChange={setNoise} />
                </>
            )}
        </div>

        <div className="md:col-span-2 space-y-6">
            <div className="grid grid-cols-2 gap-4">
                {mode === "binary" ? (
                    <>
                        <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-xl border border-green-100 dark:border-green-800">
                            <p className="text-xs font-medium text-green-600 uppercase">Precision</p>
                            <p className="text-3xl font-bold mt-1">{precision.toFixed(2)}</p>
                        </div>
                        <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-xl border border-purple-100 dark:border-purple-800">
                            <p className="text-xs font-medium text-purple-600 uppercase">Recall</p>
                            <p className="text-3xl font-bold mt-1">{recall.toFixed(2)}</p>
                        </div>
                    </>
                ) : (
                    <div className="col-span-2 bg-gray-50 dark:bg-gray-800 p-4 rounded-xl">
                        <table className="w-full text-sm text-left">
                            <thead className="text-xs text-gray-500 uppercase">
                                <tr>
                                    <th className="px-4 py-2">Class</th>
                                    <th className="px-4 py-2">Precision</th>
                                    <th className="px-4 py-2">Recall</th>
                                </tr>
                            </thead>
                            <tbody>
                                {multiMetrics.map((m) => (
                                    <tr key={m.class} className="border-t border-gray-200 dark:border-gray-700">
                                        <td className="px-4 py-2 font-medium">{m.class}</td>
                                        <td className="px-4 py-2">{m.precision.toFixed(2)}</td>
                                        <td className="px-4 py-2">{m.recall.toFixed(2)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 flex justify-center">
                <ConfusionMatrixViz 
                    matrix={mode === "binary" ? binaryMatrix : multiMatrix} 
                    labels={mode === "binary" ? ["Negative", "Positive"] : ["Cat", "Dog", "Bird"]} 
                />
            </div>
        </div>
      </div>
    </div>
  );
}
