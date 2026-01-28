"use client";

import React, { useState } from "react";
import Sidebar from "./Sidebar";
import AccuracyModule from "./modules/AccuracyModule";
import PrecisionRecallModule from "./modules/PrecisionRecallModule";
import F1SpecificityModule from "./modules/F1SpecificityModule";
import RocAucModule from "./modules/RocAucModule";
import RegressionModule from "./modules/RegressionModule";
import CorrelationModule from "./modules/CorrelationModule";
import { Menu, X } from "lucide-react";

export type ModuleType = 
  | "accuracy"
  | "precision_recall"
  | "f1_specificity"
  | "roc_auc"
  | "regression"
  | "correlation";

export default function Dashboard() {
  const [activeModule, setActiveModule] = useState<ModuleType>("accuracy");
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

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
      
      {/* Mobile Header */}
      <div className="md:hidden fixed top-0 left-0 right-0 h-16 bg-white dark:bg-gray-950 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between px-4 z-50">
        <h1 className="text-lg font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
          ML Evaluator
        </h1>
        <button 
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        >
          {isSidebarOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
        </button>
      </div>

      {/* Sidebar Overlay (Mobile) */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Sidebar Container */}
      <div className={`
        fixed inset-y-0 left-0 z-50 w-64 transform transition-transform duration-200 ease-in-out md:relative md:translate-x-0
        ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <Sidebar 
          activeModule={activeModule} 
          setActiveModule={(m) => {
            setActiveModule(m);
            setIsSidebarOpen(false); // Close on selection (mobile)
          }} 
        />
      </div>

      {/* Main Content */}
      <div className="flex-1 p-4 md:p-8 overflow-y-auto mt-16 md:mt-0">
        <div className="max-w-6xl mx-auto bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 min-h-[80vh] p-4 md:p-6">
          {renderModule()}
        </div>
      </div>
    </div>
  );
}
