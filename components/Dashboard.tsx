"use client";

import React, { useState } from "react";
import Sidebar from "./Sidebar";
import AccuracyModule from "./modules/AccuracyModule";
import PrecisionRecallModule from "./modules/PrecisionRecallModule";
import F1SpecificityModule from "./modules/F1SpecificityModule";
import RocAucModule from "./modules/RocAucModule";
import RegressionModule from "./modules/RegressionModule";
import CorrelationModule from "./modules/CorrelationModule";

export type ModuleType = 
  | "accuracy"
  | "precision_recall"
  | "f1_specificity"
  | "roc_auc"
  | "regression"
  | "correlation";

export default function Dashboard() {
  const [activeModule, setActiveModule] = useState<ModuleType>("accuracy");

  const renderModule = () => {
    switch (activeModule) {
      case "accuracy": return <AccuracyModule />;
      case "precision_recall": return <PrecisionRecallModule />;
      case "f1_specificity": return <F1SpecificityModule />;
      case "roc_auc": return <RocAucModule />;
      case "regression": return <RegressionModule />;
      case "correlation": return <CorrelationModule />;
      default: return <AccuracyModule />;
    }
  };

  return (
    <div className="flex w-full min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      <Sidebar activeModule={activeModule} setActiveModule={setActiveModule} />
      <div className="flex-1 p-8 overflow-y-auto">
        <div className="max-w-6xl mx-auto bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 min-h-[80vh] p-6">
          {renderModule()}
        </div>
      </div>
    </div>
  );
}
