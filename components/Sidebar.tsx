"use client";

import React from "react";
import { 
  Target, 
  Crosshair, 
  Scale, 
  Activity,
  TrendingUp,
  GitGraph
} from "lucide-react";
import { ModuleType } from "./Dashboard";
import clsx from "clsx";

interface SidebarProps {
  activeModule: ModuleType;
  setActiveModule: (m: ModuleType) => void;
}

export default function Sidebar({ activeModule, setActiveModule }: SidebarProps) {
  const menuItems: { id: ModuleType; label: string; icon: React.ReactNode }[] = [
    { id: "accuracy", label: "Accuracy", icon: <Target className="w-5 h-5" /> },
    { id: "precision_recall", label: "Precision & Recall", icon: <Crosshair className="w-5 h-5" /> },
    { id: "f1_specificity", label: "F1 & Specificity", icon: <Scale className="w-5 h-5" /> },
    { id: "roc_auc", label: "ROC/AUC & KS", icon: <Activity className="w-5 h-5" /> },
    { id: "regression", label: "Regression Metrics", icon: <TrendingUp className="w-5 h-5" /> },
    { id: "correlation", label: "Correlation", icon: <GitGraph className="w-5 h-5" /> },
  ];

  return (
    <aside className="w-64 bg-white dark:bg-gray-950 border-r border-gray-200 dark:border-gray-800 flex flex-col">
      <div className="p-6 border-b border-gray-200 dark:border-gray-800">
        <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
          ML Evaluator
        </h1>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Interactive Learning Tool
        </p>
      </div>
      <nav className="flex-1 p-4 space-y-1">
        {menuItems.map((item) => (
          <button
            key={item.id}
            onClick={() => setActiveModule(item.id)}
            className={clsx(
              "flex items-center w-full px-4 py-3 text-sm font-medium rounded-lg transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900",
              activeModule === item.id
                ? "bg-blue-50 text-blue-700 dark:bg-blue-900/20 dark:text-blue-400"
                : "text-gray-600 hover:bg-gray-50 dark:text-gray-400 dark:hover:bg-gray-800/50"
            )}
          >
            <span className="mr-3">{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>
      <div className="p-4 border-t border-gray-200 dark:border-gray-800">
        <div className="text-xs text-gray-500 dark:text-gray-400 text-center">
          v1.0.0 • Next.js
        </div>
      </div>
    </aside>
  );
}
