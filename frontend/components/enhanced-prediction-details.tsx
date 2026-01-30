"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { AlertTriangle, CheckCircle, TrendingUp } from "lucide-react";

interface EnhancedPredictionDetailsProps {
  prediction: string;
  probability: number;
  confidence: string;
  riskFactors: string[];
}

export function EnhancedPredictionDetails({
  prediction,
  probability,
  confidence,
  riskFactors,
}: EnhancedPredictionDetailsProps) {
  const isChurn = prediction === "Yes";
  const riskScore = isChurn ? probability : 1 - probability;

  const confidenceValue =
    confidence === "High" ? 0.85 :
      confidence === "Medium" ? 0.65 : 0.45;


  const probabilityData = [
    {
      label: "Churn Risk",
      value: probability * 100,
      fill: isChurn ? "#ef4444" : "#10b981",
    },
    {
      label: "Retention Likelihood",
      value: (1 - probability) * 100,
      fill: isChurn ? "#9ca3af" : "#3b82f6",
    },
  ];


  const gaugeData = [
    { name: "Risk", value: probability * 100, fill: "#ef4444" },
    { name: "Safe", value: (1 - probability) * 100, fill: "#10b981" },
  ];


  const confidenceData = [
    { name: "Confidence", value: confidenceValue * 100 },
    { name: "Uncertainty", value: (1 - confidenceValue) * 100 },
  ];


  const getRiskLevel = () => {
    if (probability < 0.3) return { level: "Low", color: "text-green-500", bg: "bg-green-50" };
    if (probability < 0.6) return { level: "Medium", color: "text-yellow-500", bg: "bg-yellow-50" };
    return { level: "High", color: "text-red-500", bg: "bg-red-50" };
  };

  const riskLevel = getRiskLevel();

  return (
    <div className="space-y-6">

      <Card className="border-border bg-card overflow-hidden">
        <div
          className={`h-2 w-full ${isChurn ? "bg-red-500" : "bg-green-500"}`}
        />
        <CardHeader>
          <CardTitle className="text-xl flex items-center gap-2">
            {isChurn ? (
              <>
                <AlertTriangle className="h-6 w-6 text-red-500" />
                High Churn Risk Detected
              </>
            ) : (
              <>
                <CheckCircle className="h-6 w-6 text-green-500" />
                Low Churn Risk
              </>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">

          <div className="grid grid-cols-3 gap-4">
            <div className="rounded-lg bg-secondary/50 p-4 text-center">
              <p className="text-sm text-muted-foreground mb-2">Prediction</p>
              <p className={`text-3xl font-bold ${isChurn ? "text-red-500" : "text-green-500"}`}>
                {prediction}
              </p>
            </div>
            <div className="rounded-lg bg-secondary/50 p-4 text-center">
              <p className="text-sm text-muted-foreground mb-2">Confidence</p>
              <p className="text-3xl font-bold text-blue-500">{confidence}</p>
              <p className="text-xs text-muted-foreground mt-1">{(confidenceValue * 100).toFixed(0)}%</p>
            </div>
            <div className={`rounded-lg p-4 text-center ${riskLevel.bg}`}>
              <p className="text-sm text-muted-foreground mb-2">Risk Level</p>
              <p className={`text-3xl font-bold ${riskLevel.color}`}>
                {riskLevel.level}
              </p>
            </div>
          </div>


          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold">Churn Probability</span>
              <span className="text-sm font-mono font-bold">{(probability * 100).toFixed(1)}%</span>
            </div>
            <div className="flex gap-2">
              <div className="flex-1 rounded-full overflow-hidden bg-secondary h-8">
                <div
                  className={`h-full rounded-full transition-all flex items-center justify-end pr-3 ${isChurn ? "bg-gradient-to-r from-red-400 to-red-600" : "bg-gradient-to-r from-green-400 to-green-600"
                    }`}
                  style={{ width: `${probability * 100}%` }}
                >
                  {probability > 0.1 && (
                    <span className="text-white text-xs font-bold">{(probability * 100).toFixed(0)}%</span>
                  )}
                </div>
              </div>
            </div>
          </div>


          <div className="grid grid-cols-2 gap-4">
            <div className="rounded-lg border border-border p-4">
              <p className="text-sm font-semibold mb-3">Probability Distribution</p>
              <ResponsiveContainer width="100%" height={150}>
                <PieChart>
                  <Pie
                    data={probabilityData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={60}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {probabilityData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => {
                    if (typeof value === "number") return `${value.toFixed(1)}%`;
                    const num = Number(value);
                    return isNaN(num) ? String(value) : `${num.toFixed(1)}%`;
                  }} />
                </PieChart>
              </ResponsiveContainer>
              <div className="grid grid-cols-2 gap-2 mt-3 text-xs">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: probabilityData[0].fill }} />
                  <span>Churn: {(probability * 100).toFixed(1)}%</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: probabilityData[1].fill }} />
                  <span>Retain: {((1 - probability) * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            {/* Confidence Gauge */}
            <div className="rounded-lg border border-border p-4">
              <p className="text-sm font-semibold mb-3">Model Confidence</p>
              <ResponsiveContainer width="100%" height={150}>
                <PieChart>
                  <Pie
                    data={[
                      { name: "Confidence", value: confidenceValue * 100 },
                      { name: "Uncertainty", value: (1 - confidenceValue) * 100 },
                    ]}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={60}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    <Cell fill="#3b82f6" />
                    <Cell fill="#e5e7eb" />
                  </Pie>
                  <Tooltip formatter={(value) => {
                    if (typeof value === "number") return `${value.toFixed(1)}%`;
                    const num = Number(value);
                    return isNaN(num) ? String(value) : `${num.toFixed(1)}%`;
                  }} />
                </PieChart>
              </ResponsiveContainer>
              <div className="mt-3 text-xs text-center">
                <p className="font-semibold text-blue-500">{(confidenceValue * 100).toFixed(0)}% Confident</p>
                <p className="text-muted-foreground">in this prediction</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Risk Factors Analysis */}
      {riskFactors && riskFactors.length > 0 && (
        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Identified Risk Factors ({riskFactors.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {riskFactors.map((factor, idx) => (
                <div
                  key={idx}
                  className="flex items-start gap-3 p-3 rounded-lg bg-red-50 border border-red-200"
                >
                  <AlertTriangle className="h-5 w-5 text-red-500 mt-0.5 shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-red-900">{factor}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Prediction Insights */}
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-lg">Prediction Insights</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 rounded-lg bg-blue-50 border border-blue-200">
              <p className="text-xs text-blue-600 font-semibold mb-2">PROBABILITY SCORE</p>
              <p className="text-2xl font-bold text-blue-600">{(probability * 100).toFixed(1)}%</p>
              <p className="text-xs text-blue-600 mt-1">
                {isChurn ? "High likelihood of churn" : "Low likelihood of churn"}
              </p>
            </div>
            <div className="p-4 rounded-lg bg-purple-50 border border-purple-200">
              <p className="text-xs text-purple-600 font-semibold mb-2">CONFIDENCE LEVEL</p>
              <p className="text-2xl font-bold text-purple-600">{(confidenceValue * 100).toFixed(0)}%</p>
              <p className="text-xs text-purple-600 mt-1">
                Model confidence in prediction
              </p>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200">
            <p className="text-sm font-semibold mb-2">What This Means:</p>
            <p className="text-sm text-gray-700 leading-relaxed">
              {isChurn
                ? `This customer has a ${(probability * 100).toFixed(0)}% probability of churning. With ${confidence.toLowerCase()} confidence, 
                   our model indicates significant churn risk. Focus on retention strategies immediately.`
                : `This customer has a ${((1 - probability) * 100).toFixed(0)}% probability of staying. With ${confidence.toLowerCase()} confidence, 
                   the retention outlook is positive. Consider engagement opportunities.`}
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
