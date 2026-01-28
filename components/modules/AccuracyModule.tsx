"use client";

import React, { useState, useMemo } from "react";
import SliderControl from "../ui/SliderControl";
import ConfusionMatrixViz from "../ui/ConfusionMatrixViz";

export default function AccuracyModule() {
  const [mode, setMode] = useState<"binary" | "multi">("binary");
  
  // Binary State
  const [tp, setTp] = useState(50);
  const [tn, setTn] = useState(50);
  const [fp, setFp] = useState(10);
  const [fn, setFn] = useState(10);

  // Multi-class State
  const [samples, setSamples] = useState(300);
  const [noise, setNoise] = useState(20); // % error

  const binaryMatrix = useMemo(() => [[tn, fp], [fn, tp]], [tp, tn, fp, fn]);
  const binaryAccuracy = (tp + tn) / (tp + tn + fp + fn);

  const multiMatrix = useMemo(() => {
    // Deterministic simulation based on noise
    const classes = 3;
    const perClass = Math.floor(samples / classes);
    const matrix = Array(classes).fill(0).map(() => Array(classes).fill(0));
    
    // Distribute
    for(let i=0; i<classes; i++) {
        const correct = Math.floor(perClass * (1 - noise / 100));
        const error = perClass - correct;
        
        matrix[i][i] = correct;
        // Distribute error to other classes
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

  const multiAccuracy = useMemo(() => {
    let correct = 0;
    let total = 0;
    for(let i=0; i<multiMatrix.length; i++) {
        for(let j=0; j<multiMatrix.length; j++) {
            if(i === j) correct += multiMatrix[i][j];
            total += multiMatrix[i][j];
        }
    }
    return total > 0 ? correct / total : 0;
  }, [multiMatrix]);

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold mb-2">Accuracy</h2>
        <p className="text-gray-600 dark:text-gray-400">
            The ratio of correct predictions (True Positives + True Negatives) to the total number of predictions.
        </p>
      </div>

      <div className="flex space-x-4 bg-gray-100 dark:bg-gray-800 p-1 rounded-lg w-fit">
        <button
            onClick={() => setMode("binary")}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${mode === 'binary' ? 'bg-white shadow text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
        >
            Binary Classification
        </button>
        <button
            onClick={() => setMode("multi")}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${mode === 'multi' ? 'bg-white shadow text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
        >
            Multi-class (3 Classes)
        </button>
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
            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-100 dark:border-blue-800">
                <div className="flex justify-between items-end">
                    <div>
                        <p className="text-sm font-medium text-blue-600 dark:text-blue-400 uppercase tracking-wider">Calculated Accuracy</p>
                        <p className="text-4xl font-bold text-gray-900 dark:text-white mt-2">
                            {(mode === "binary" ? binaryAccuracy : multiAccuracy).toLocaleString(undefined, {style: 'percent', minimumFractionDigits: 2})}
                        </p>
                    </div>
                    <div className="text-right">
                        <div className="text-sm font-mono bg-white dark:bg-gray-900 px-3 py-1 rounded border border-blue-200 dark:border-blue-700">
                             {mode === "binary" ? "ACC = (TP + TN) / Total" : "ACC = Trace(Matrix) / Total"}
                        </div>
                    </div>
                </div>
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
