export function generateMultiClassMatrix(totalSamples: number, noisePercentage: number, classCount: number = 3) {
  // 1. Distribute samples accurately to ensure sum equals totalSamples
  // integer division flooring can lose samples (e.g. 100 / 3 = 33, sum=99).
  const basePerClass = Math.floor(totalSamples / classCount);
  const remainderSamples = totalSamples % classCount;
  
  const matrix = Array(classCount).fill(0).map(() => Array(classCount).fill(0));

  for (let i = 0; i < classCount; i++) {
    // Add extra sample to first 'remainderSamples' classes
    const samplesForThisClass = basePerClass + (i < remainderSamples ? 1 : 0);
    
    // 2. Apply noise
    // Correct predictions (diagonal)
    const correct = Math.floor(samplesForThisClass * (1 - noisePercentage / 100));
    const error = samplesForThisClass - correct;
    
    matrix[i][i] = correct;
    
    // Distribute error to other classes (off-diagonal)
    const otherClassesCount = classCount - 1;
    if (otherClassesCount > 0) {
        const errorPerClass = Math.floor(error / otherClassesCount);
        let errorRemainder = error % otherClassesCount;
        
        for (let j = 0; j < classCount; j++) {
            if (i !== j) {
                // Add base error + 1 if we have remainder
                matrix[i][j] = errorPerClass + (errorRemainder > 0 ? 1 : 0);
                errorRemainder--;
            }
        }
    }
  }
  
  return matrix;
}
