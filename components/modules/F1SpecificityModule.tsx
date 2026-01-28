"use client";

import React, { useState, useMemo } from "react";
import SliderControl from "../ui/SliderControl";
import ConfusionMatrixViz from "../ui/ConfusionMatrixViz";

export default function F1SpecificityModule() {
  const [mode, setMode] = useState<"binary" | "multi">("binary");
  
  const [tp, setTp] = useState(50);
  const [tn, setTn] = useState(50);
  const [fp, setFp] = useState(20);
  const [fn, setFn] = useState(20);

  const [samples, setSamples] = useState(300);
  const [noise, setNoise] = useState(20);

  const binaryMatrix = useMemo(() => [[tn, fp], [fn, tp]], [tp, tn, fp, fn]);
  
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1 = 2 * (precision * recall) / (precision + recall) || 0;
  const specificity = tn / (tn + fp) || 0;

  const multiMatrix = useMemo(() => {
    const classes = 3;
    const perClass = Math.floor(samples / classes);
    const matrix = Array(classes).fill(0).map(() => Array(classes).fill(0));
    for(let i=0; i<classes; i++) {
        const correct = Math.floor(perClass * (1 - noise / 100));
        const error = perClass - correct;
        matrix[i][i] = correct;
        const errorPerClass = Math.floor(error / (classes - 1));
        let remainder = error % (classes - 1);
        for(let j=0; j<classes; j++) {
            if (i !== j) {
                matrix[i][j] = errorPerClass + (remainder > 0 ? 1 : 0);
                remainder--;
            }
        }
    }
    return matrix;
  }, [samples, noise]);

  const multiMetrics = useMemo(() => {
      let macroF1 = 0;
      const metrics = [];
      const total = multiMatrix.flat().reduce((a,b)=>a+b, 0);

      for(let i=0; i<3; i++) {
          const tp = multiMatrix[i][i];
          const fp = multiMatrix.reduce((acc, row) => acc + row[i], 0) - tp;
          const fn = multiMatrix[i].reduce((acc, val) => acc + val, 0) - tp;
          // TN for class i = Total - (TP+FP+FN)
          const tn = total - (tp + fp + fn);

          const p = tp / (tp + fp) || 0;
          const r = tp / (tp + fn) || 0;
          const f = 2*(p*r)/(p+r) || 0;
          const s = tn / (tn + fp) || 0;
          
          macroF1 += f;
          metrics.push({ class: ["Cat", "Dog", "Bird"][i], f1: f, specificity: s });
      }
      return { metrics, macroF1: macroF1 / 3 };
  }, [multiMatrix]);

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold mb-2">F1-Score & Specificity</h2>
        <div className="grid grid-cols-2 gap-4 text-sm text-gray-600 dark:text-gray-400">
            <p><strong>F1-Score:</strong> Harmonic mean of Precision and Recall.</p>
            <p><strong>Specificity:</strong> True Negative Rate (TN / TN+FP).</p>
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
                    <SliderControl label="True Negatives (TN)" value={tn} min={0} max={100} onChange={setTn} />
                    <SliderControl label="False Positives (FP)" value={fp} min={0} max={100} onChange={setFp} />
                    <SliderControl label="False Negatives (FN)" value={fn} min={0} max={100} onChange={setFn} />
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
                        <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-xl border border-orange-100 dark:border-orange-800">
                            <p className="text-xs font-medium text-orange-600 uppercase">F1-Score</p>
                            <p className="text-3xl font-bold mt-1">{f1.toFixed(2)}</p>
                        </div>
                        <div className="bg-teal-50 dark:bg-teal-900/20 p-4 rounded-xl border border-teal-100 dark:border-teal-800">
                            <p className="text-xs font-medium text-teal-600 uppercase">Specificity</p>
                            <p className="text-3xl font-bold mt-1">{specificity.toFixed(2)}</p>
                        </div>
                    </>
                ) : (
                    <div className="col-span-2 space-y-4">
                        <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-xl border border-blue-100 dark:border-blue-800">
                             <p className="text-xs font-medium text-blue-600 uppercase">Macro F1-Score</p>
                             <p className="text-3xl font-bold mt-1">{multiMetrics.macroF1.toFixed(2)}</p>
                        </div>
                        <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-xl">
                            <table className="w-full text-sm text-left">
                                <thead className="text-xs text-gray-500 uppercase">
                                    <tr>
                                        <th className="px-4 py-2">Class</th>
                                        <th className="px-4 py-2">F1-Score</th>
                                        <th className="px-4 py-2">Specificity</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {multiMetrics.metrics.map((m) => (
                                        <tr key={m.class} className="border-t border-gray-200 dark:border-gray-700">
                                            <td className="px-4 py-2 font-medium">{m.class}</td>
                                            <td className="px-4 py-2">{m.f1.toFixed(2)}</td>
                                            <td className="px-4 py-2">{m.specificity.toFixed(2)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
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
