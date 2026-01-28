import React from "react";
import clsx from "clsx";

interface ConfusionMatrixProps {
  matrix: number[][]; // 2x2 or 3x3
  labels: string[];
}

export default function ConfusionMatrixViz({ matrix, labels }: ConfusionMatrixProps) {
  const maxValue = Math.max(...matrix.flat());

  const getColor = (value: number) => {
    // Simple blue scale based on intensity
    const intensity = maxValue > 0 ? value / maxValue : 0;
    // Tailwind blue-600 is roughly rgb(37, 99, 235)
    // We'll use rgba for simple opacity scaling
    return `rgba(37, 99, 235, ${Math.max(0.1, intensity)})`;
  };

  return (
    <div className="flex flex-col items-center">
      <h3 className="text-sm font-semibold text-gray-500 mb-2 uppercase tracking-wider">
        Predicted Class
      </h3>
      <div className="flex">
        <div className="flex items-center justify-center mr-2">
          <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider -rotate-90 whitespace-nowrap">
            Actual Class
          </h3>
        </div>
        
        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${labels.length}, minmax(80px, 1fr))` }}>
            {/* Header Row */}
            {labels.map((label, i) => (
              <div key={`head-${i}`} className="text-center text-xs font-bold text-gray-600 dark:text-gray-400 p-2">
                {label}
              </div>
            ))}
            
            {/* Matrix Cells */}
            {matrix.map((row, i) => (
              <React.Fragment key={`row-${i}`}>
                {row.map((val, j) => (
                  <div
                    key={`cell-${i}-${j}`}
                    className="flex flex-col items-center justify-center p-4 rounded-md border border-gray-100 dark:border-gray-700 transition-all"
                    style={{ backgroundColor: getColor(val) }}
                    title={`Actual: ${labels[i]}, Predicted: ${labels[j]}`}
                  >
                    <span className="text-lg font-bold text-gray-900 dark:text-gray-100 mix-blend-multiply dark:mix-blend-screen">
                      {val}
                    </span>
                  </div>
                ))}
              </React.Fragment>
            ))}
        </div>
      </div>
      
      {/* Side Labels for Rows (kind of tricky in grid, usually done with absolute or flex wrappers. Keeping it simple above with the rotated text) */}
    </div>
  );
}
