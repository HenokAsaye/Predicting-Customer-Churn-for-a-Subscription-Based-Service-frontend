"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { CustomerForm } from "@/components/customer-form";
import { PredictionResult } from "@/components/prediction-result";
import { ModelMetrics } from "@/components/model-metrics";
import { ModelMetricsBackend } from "@/components/model-metrics-backend";
import { FeatureImportance } from "@/components/feature-importance";
import { PredictionExplanation } from "@/components/prediction-explanation";
import { BatchPrediction } from "@/components/batch-prediction";
import { EnhancedPredictionDetails } from "@/components/enhanced-prediction-details";
import {
  BrainCircuit,
  RotateCcw,
  Zap,
  BarChart3,
  FileText,
  AlertTriangle,
  CheckCircle,
} from "lucide-react";
import { api, type CustomerData, type PredictionResponse, type FeatureImportanceItem } from "@/lib/api";

const defaultCustomerData: Partial<CustomerData> = {
  gender: "Male",
  SeniorCitizen: 0,
  Partner: "No",
  Dependents: "No",
  tenure: 12,
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "DSL",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "No",
  StreamingMovies: "No",
  Contract: "Month-to-month",
  PaperlessBilling: "No",
  PaymentMethod: "Credit card (automatic)",
  MonthlyCharges: 65.0,
  TotalCharges: 780.0,
};

export default function ChurnPredictionPage() {
  const [customerData, setCustomerData] =
    useState<Partial<CustomerData>>(defaultCustomerData);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("input");
  const [modelMetrics, setModelMetrics] = useState<any>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportanceItem[]>([]);
  const [apiError, setApiError] = useState<string | null>(null);
  const [apiHealthy, setApiHealthy] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await api.checkHealth();
        setApiHealthy(health.status === "healthy" && health.model_loaded);
        if (!health.model_loaded) {
          setApiError("Model not loaded. Please train the model first.");
        }
      } catch (error) {
        setApiHealthy(false);
        setApiError("Cannot connect to API. Make sure the backend is running on http://localhost:8000");
      }
    };

    checkHealth();
  }, []);

  // Load model info and feature importance
  useEffect(() => {
    const loadModelData = async () => {
      try {
        const [metrics, features] = await Promise.all([
          api.getModelInfo().catch(() => null),
          api.getFeatureImportance().catch(() => []),
        ]);

        if (metrics) {
          setModelMetrics(metrics.metrics);
        }
        if (features) {
          setFeatureImportance(features);
        }
      } catch (error) {
        console.error("Error loading model data:", error);
      }
    };

    if (apiHealthy) {
      loadModelData();
    }
  }, [apiHealthy]);

  // Make prediction using API
  const runPrediction = useCallback(async () => {
    if (!apiHealthy || !customerData) {
      setApiError("API not ready or customer data incomplete");
      return;
    }

    setIsLoading(true);
    setApiError(null);

    try {
      const result = await api.predict(customerData as CustomerData);
      setPrediction(result);
      setActiveTab("prediction");
    } catch (error) {
      setApiError(error instanceof Error ? error.message : "Prediction failed");
    } finally {
      setIsLoading(false);
    }
  }, [customerData, apiHealthy]);

  const resetForm = useCallback(() => {
    setCustomerData(defaultCustomerData);
    setPrediction(null);
    setApiError(null);
    setActiveTab("input");
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {/* API Error Alert */}
      {apiError && (
        <Alert className="mx-auto max-w-7xl mt-4 border-destructive bg-destructive/10">
          <AlertTriangle className="h-4 w-4 text-destructive" />
          <AlertDescription className="text-destructive">{apiError}</AlertDescription>
        </Alert>
      )}

      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                <BrainCircuit className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-foreground">ChurnPredict</h1>
                <p className="text-sm text-muted-foreground flex items-center gap-2">
                  AI-Powered Customer Retention
                  {apiHealthy && (
                    <>
                      <span>•</span>
                      <span className="flex items-center gap-1">
                        <CheckCircle className="h-3 w-3 text-green-500" />
                        Connected
                      </span>
                    </>
                  )}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={resetForm}
                className="gap-2 bg-transparent"
              >
                <RotateCcw className="h-4 w-4" />
                Reset
              </Button>
              <Button
                size="sm"
                onClick={runPrediction}
                disabled={isLoading || !apiHealthy}
                className="gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
              >
                <Zap className="h-4 w-4" />
                {isLoading ? "Predicting..." : "Predict"}
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-5 bg-secondary/50">
            <TabsTrigger value="input" className="gap-2">
              <FileText className="h-4 w-4" />
              <span className="hidden sm:inline">Input Data</span>
            </TabsTrigger>
            <TabsTrigger value="prediction" className="gap-2">
              <Zap className="h-4 w-4" />
              <span className="hidden sm:inline">Prediction</span>
            </TabsTrigger>
            <TabsTrigger value="batch" className="gap-2">
              <BarChart3 className="h-4 w-4" />
              <span className="hidden sm:inline">Batch</span>
            </TabsTrigger>
            <TabsTrigger value="metrics" className="gap-2">
              <BarChart3 className="h-4 w-4" />
              <span className="hidden sm:inline">Metrics</span>
            </TabsTrigger>
            <TabsTrigger value="explanation" className="gap-2">
              <BrainCircuit className="h-4 w-4" />
              <span className="hidden sm:inline">Explanation</span>
            </TabsTrigger>
          </TabsList>

          {/* Input Tab */}
          <TabsContent value="input" className="space-y-6">
            <Card className="border-border bg-card">
              <CardHeader>
                <CardTitle className="text-xl">Customer Information</CardTitle>
                <p className="text-sm text-muted-foreground">
                  Enter customer details to predict churn probability
                </p>
              </CardHeader>
              <CardContent>
                <CustomerForm data={customerData} onChange={setCustomerData} />
              </CardContent>
            </Card>
            <div className="flex justify-end gap-3">
              <Button variant="outline" onClick={resetForm} className="gap-2 bg-transparent">
                <RotateCcw className="h-4 w-4" />
                Reset Form
              </Button>
              <Button
                onClick={runPrediction}
                disabled={isLoading || !apiHealthy}
                className="gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
              >
                <Zap className="h-4 w-4" />
                {isLoading ? "Running Prediction..." : "Run Prediction"}
              </Button>
            </div>
          </TabsContent>

          {/* Prediction Tab */}
          <TabsContent value="prediction" className="space-y-6">
            {prediction !== null ? (
              <div className="space-y-6">
                <EnhancedPredictionDetails
                  prediction={prediction.prediction}
                  probability={prediction.probability}
                  confidence={prediction.confidence}
                  riskFactors={prediction.risk_factors || []}
                />
                {featureImportance && featureImportance.length > 0 && (
                  <FeatureImportance
                    features={featureImportance.map(f => ({
                      name: f.feature,
                      importance: f.importance
                    }))}
                  />
                )}
              </div>
            ) : (
              <Card className="border-border bg-card">
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                    <Zap className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <h3 className="mt-4 text-lg font-semibold text-foreground">
                    No Prediction Yet
                  </h3>
                  <p className="mt-2 text-center text-sm text-muted-foreground max-w-sm">
                    Enter customer data in the Input tab and click
                    &quot;Predict&quot; to see the churn prediction results.
                  </p>
                  <Button
                    className="mt-6 gap-2"
                    onClick={() => setActiveTab("input")}
                  >
                    <FileText className="h-4 w-4" />
                    Go to Input
                  </Button>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Batch Prediction Tab */}
          <TabsContent value="batch" className="space-y-6">
            <BatchPrediction />
          </TabsContent>

          {/* Metrics Tab */}
          <TabsContent value="metrics" className="space-y-6">
            <ModelMetricsBackend refreshTrigger={prediction !== null} />
          </TabsContent>

          {/* Explanation Tab */}
          <TabsContent value="explanation" className="space-y-6">
            {prediction !== null ? (
              <div className="grid gap-6 lg:grid-cols-2">
                <Card className="border-border bg-card">
                  <CardHeader>
                    <CardTitle className="text-lg">Risk Factors</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {prediction.risk_factors && prediction.risk_factors.length > 0 ? (
                      <ul className="space-y-3">
                        {prediction.risk_factors.map((factor, idx) => (
                          <li key={idx} className="flex items-start gap-3">
                            <AlertTriangle className="h-5 w-5 text-destructive mt-0.5 shrink-0" />
                            <span className="text-sm text-foreground">{factor}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-sm text-muted-foreground">No significant risk factors identified.</p>
                    )}
                  </CardContent>
                </Card>
                <Card className="border-border bg-card">
                  <CardHeader>
                    <CardTitle className="text-lg">
                      Retention Recommendations
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {prediction.prediction === "Yes" ? (
                      <>
                        <p className="text-sm text-muted-foreground">
                          Based on this customer&apos;s profile, consider the
                          following retention strategies:
                        </p>
                        <ul className="space-y-2">
                          <li className="flex items-start gap-2 text-sm">
                            <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-primary shrink-0" />
                            <span className="text-foreground">
                              Offer a loyalty discount for upgrading to an annual plan
                            </span>
                          </li>
                          <li className="flex items-start gap-2 text-sm">
                            <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-primary shrink-0" />
                            <span className="text-foreground">
                              Schedule a customer success check-in call
                            </span>
                          </li>
                          <li className="flex items-start gap-2 text-sm">
                            <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-primary shrink-0" />
                            <span className="text-foreground">
                              Provide personalized feature recommendations
                            </span>
                          </li>
                          <li className="flex items-start gap-2 text-sm">
                            <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-primary shrink-0" />
                            <span className="text-foreground">
                              Send engagement-focused email campaigns
                            </span>
                          </li>
                        </ul>
                      </>
                    ) : (
                      <>
                        <p className="text-sm text-muted-foreground">
                          This customer shows positive retention indicators.
                          Consider these engagement strategies:
                        </p>
                        <ul className="space-y-2">
                          <li className="flex items-start gap-2 text-sm">
                            <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-accent shrink-0" />
                            <span className="text-foreground">
                              Invite them to a customer advocacy program
                            </span>
                          </li>
                          <li className="flex items-start gap-2 text-sm">
                            <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-accent shrink-0" />
                            <span className="text-foreground">
                              Offer early access to new features
                            </span>
                          </li>
                          <li className="flex items-start gap-2 text-sm">
                            <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-accent shrink-0" />
                            <span className="text-foreground">
                              Request feedback for product improvement
                            </span>
                          </li>
                          <li className="flex items-start gap-2 text-sm">
                            <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-accent shrink-0" />
                            <span className="text-foreground">
                              Consider for upsell opportunities
                            </span>
                          </li>
                        </ul>
                      </>
                    )}
                  </CardContent>
                </Card>
              </div>
            ) : (
              <Card className="border-border bg-card">
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                    <BrainCircuit className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <h3 className="mt-4 text-lg font-semibold text-foreground">
                    No Explanation Available
                  </h3>
                  <p className="mt-2 text-center text-sm text-muted-foreground max-w-sm">
                    Run a prediction first to see detailed explanations of the
                    factors influencing the result.
                  </p>
                  <Button
                    className="mt-6 gap-2"
                    onClick={() => setActiveTab("input")}
                  >
                    <FileText className="h-4 w-4" />
                    Go to Input
                  </Button>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t border-border bg-card/50 mt-12">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-muted-foreground">
            ChurnPredict - Machine Learning Lab Project | Customer Churn
            Prediction System
          </p>
        </div>
      </footer>
    </div>
  );
}
