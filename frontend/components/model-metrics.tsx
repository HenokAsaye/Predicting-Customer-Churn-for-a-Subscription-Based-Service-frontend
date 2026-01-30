"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Cell,
  Tooltip,
} from "recharts";

interface ModelMetricsProps {
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
  };
}

export function ModelMetrics({ metrics }: ModelMetricsProps) {
  const mappedMetrics = {
    accuracy: metrics.accuracy,
    precision: metrics.precision,
    recall: metrics.recall,
    f1Score: metrics.f1Score,
  };

  const data = [
    { name: "Accuracy", value: mappedMetrics.accuracy * 100, fill: "var(--color-chart-1)" },
    { name: "Precision", value: mappedMetrics.precision * 100, fill: "var(--color-chart-2)" },
    { name: "Recall", value: mappedMetrics.recall * 100, fill: "var(--color-chart-3)" },
    { name: "F1 Score", value: mappedMetrics.f1Score * 100, fill: "var(--color-chart-1)" },
  ];

  return (
    <Card className="border-border bg-card">
      <CardHeader>
        <CardTitle className="text-lg">Model Performance Metrics</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[200px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={data}
              layout="vertical"
              margin={{ top: 0, right: 30, left: 0, bottom: 0 }}
            >
              <XAxis
                type="number"
                domain={[0, 100]}
                tickFormatter={(value) => `${value}%`}
                tick={{ fill: "var(--color-muted-foreground)", fontSize: 12 }}
                axisLine={{ stroke: "var(--color-border)" }}
                tickLine={{ stroke: "var(--color-border)" }}
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fill: "var(--color-foreground)", fontSize: 12 }}
                axisLine={false}
                tickLine={false}
                width={80}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="rounded-lg border border-border bg-popover px-3 py-2 shadow-md">
                        <p className="text-sm font-medium text-popover-foreground">
                          {payload[0].payload.name}: {(() => {
                            const v = payload[0].value;
                            if (typeof v === "number") return `${v.toFixed(1)}%`;
                            const num = Number(v);
                            return isNaN(num) ? String(v) : `${num.toFixed(1)}%`;
                          })()}
                        </p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 grid grid-cols-2 gap-3">
          {data.map((metric) => (
            <div
              key={metric.name}
              className="flex items-center justify-between rounded-lg bg-secondary/50 px-3 py-2"
            >
              <span className="text-sm text-muted-foreground">{metric.name}</span>
              <span className="font-mono font-semibold text-foreground">
                {(() => {
                  const v = metric.value;
                  if (typeof v === "number") return v.toFixed(1) + "%";
                  const num = Number(v);
                  return isNaN(num) ? String(v) : num.toFixed(1) + "%";
                })()}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
