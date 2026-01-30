"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { CheckCircle, AlertTriangle, Upload } from "lucide-react";
import { api, type CustomerData, type PredictionResponse } from "@/lib/api";

interface BatchPredictionProps {
  onResults?: (results: PredictionResponse[]) => void;
}

export function BatchPrediction({ onResults }: BatchPredictionProps) {
  const [csvData, setCsvData] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<Array<PredictionResponse & { row: number }>>(
    []
  );
  const [error, setError] = useState<string | null>(null);
  const [successCount, setSuccessCount] = useState(0);

  const parseCsvData = (csv: string): Partial<CustomerData>[] => {
    const lines = csv.trim().split("\n");
    if (lines.length < 2) {
      throw new Error("CSV must have header and at least one data row");
    }

    const headers = lines[0].split(",").map((h) => h.trim());
    const data: Partial<CustomerData>[] = [];

    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(",").map((v) => v.trim());
      const row: any = {};

      headers.forEach((header, index) => {
        const value = values[index];
        // Convert numeric values
        if (
          [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "SeniorCitizen",
          ].includes(header)
        ) {
          row[header] = isNaN(parseFloat(value))
            ? value
            : parseFloat(value);
        } else {
          row[header] = value;
        }
      });

      data.push(row);
    }

    return data;
  };

  const handlePredictBatch = async () => {
    setError(null);
    setResults([]);
    setSuccessCount(0);

    try {
      if (!csvData.trim()) {
        setError("Please enter CSV data");
        return;
      }

      setIsLoading(true);
      const customers = parseCsvData(csvData);

      const batchResults = await api.predictBatch(
        customers as CustomerData[]
      );

      const resultsWithRows = batchResults.predictions.map((pred, idx) => ({
        ...pred,
        row: idx + 2,
      }));

      setResults(resultsWithRows);
      setSuccessCount(
        resultsWithRows.filter((r) => r.prediction === "No").length
      );

      if (onResults) {
        onResults(batchResults.predictions);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Batch prediction failed");
    } finally {
      setIsLoading(false);
    }
  };

  const downloadResults = () => {
    if (results.length === 0) return;

    const headers = [
      "Row",
      "Prediction",
      "Probability",
      "Confidence",
      "Risk Factors",
    ];
    const rows = results.map((r) => [
      r.row,
      r.prediction,
      (r.probability * 100).toFixed(2) + "%",
      r.confidence,
      r.risk_factors?.join("; ") || "None",
    ]);

    const csv = [headers, ...rows].map((row) => row.map((cell) => `"${cell}"`).join(",")).join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `churn_predictions_${new Date().toISOString().split("T")[0]}.csv`;
    link.click();
  };

  return (
    <div className="space-y-6">
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-lg">CSV Data Format</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground mb-4">
            Upload customer data in CSV format with the following columns:
          </p>
          <div className="bg-secondary/50 p-4 rounded-lg text-sm font-mono text-muted-foreground overflow-x-auto">
            <pre>
              {`gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
Male,0,Yes,No,12,Yes,No,DSL,Yes,No,No,No,No,No,Month-to-month,No,Credit card (automatic),65.0,780.0`}
            </pre>
          </div>
        </CardContent>
      </Card>

      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-lg">Batch Prediction Input</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {error && (
            <Alert className="border-destructive bg-destructive/10">
              <AlertTriangle className="h-4 w-4 text-destructive" />
              <AlertDescription className="text-destructive">
                {error}
              </AlertDescription>
            </Alert>
          )}

          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              CSV Data
            </label>
            <Textarea
              placeholder="Paste your CSV data here..."
              value={csvData}
              onChange={(e) => setCsvData(e.target.value)}
              className="min-h-[200px] bg-secondary border-border font-mono text-sm"
            />
          </div>

          <div className="flex gap-3">
            <Button
              onClick={handlePredictBatch}
              disabled={isLoading || !csvData.trim()}
              className="gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
            >
              <Upload className="h-4 w-4" />
              {isLoading ? "Processing..." : "Process Batch"}
            </Button>
            {results.length > 0 && (
              <Button
                variant="outline"
                onClick={downloadResults}
                className="gap-2 bg-transparent"
              >
                Download Results
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {results.length > 0 && (
        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-lg">
              Results ({results.length} customers)
            </CardTitle>
            <div className="flex gap-4 mt-2">
              <div className="text-sm">
                <p className="text-muted-foreground">Low Risk (No Churn)</p>
                <p className="text-2xl font-bold text-accent">
                  {successCount}
                </p>
              </div>
              <div className="text-sm">
                <p className="text-muted-foreground">High Risk (Churn)</p>
                <p className="text-2xl font-bold text-destructive">
                  {results.length - successCount}
                </p>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-border">
                  <tr>
                    <th className="text-left py-2 px-4 font-medium">Row</th>
                    <th className="text-left py-2 px-4 font-medium">
                      Prediction
                    </th>
                    <th className="text-left py-2 px-4 font-medium">
                      Probability
                    </th>
                    <th className="text-left py-2 px-4 font-medium">
                      Confidence
                    </th>
                    <th className="text-left py-2 px-4 font-medium">
                      Risk Factors
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((result, idx) => (
                    <tr
                      key={idx}
                      className="border-b border-border hover:bg-secondary/50 transition-colors"
                    >
                      <td className="py-2 px-4">{result.row}</td>
                      <td className="py-2 px-4">
                        <div className="flex items-center gap-2">
                          {result.prediction === "Yes" ? (
                            <>
                              <AlertTriangle className="h-4 w-4 text-destructive" />
                              <span className="text-destructive font-medium">
                                {result.prediction}
                              </span>
                            </>
                          ) : (
                            <>
                              <CheckCircle className="h-4 w-4 text-accent" />
                              <span className="text-accent font-medium">
                                {result.prediction}
                              </span>
                            </>
                          )}
                        </div>
                      </td>
                      <td className="py-2 px-4">
                        {(result.probability * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 px-4">{result.confidence}</td>
                      <td className="py-2 px-4 text-xs text-muted-foreground">
                        {result.risk_factors?.length
                          ? result.risk_factors[0]
                          : "None"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
