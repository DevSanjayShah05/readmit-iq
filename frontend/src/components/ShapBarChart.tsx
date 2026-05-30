/**
 * ShapBarChart — horizontal bar chart of SHAP feature contributions.
 *
 * Each bar's length represents the magnitude of one feature's contribution
 * to the prediction. Bars on the right (positive shap_value) pushed the
 * probability up; bars on the left (negative shap_value) pulled it down.
 *
 * We show the top N features by absolute magnitude, since a model with
 * 19 features would otherwise produce a noisy chart dominated by near-zero
 * contributions.
 */
import { useMemo } from "react";
import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ExplanationResponse } from "../types/api";


interface Props {
  explanation: ExplanationResponse;
  topN?: number;
}


export function ShapBarChart({ explanation, topN = 8 }: Props) {
  // Sort contributions by absolute magnitude; take the top N; reverse so
  // the largest appears at the top of a horizontal chart.
  const data = useMemo(() => {
    return [...explanation.contributions]
      .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
      .slice(0, topN)
      .reverse()
      .map((c) => ({
        feature: c.feature_name,
        shap_value: c.shap_value,
        feature_value: c.feature_value,
      }));
  }, [explanation, topN]);

  const baseline = (explanation.baseline_probability * 100).toFixed(1);
  const predicted = (explanation.predicted_probability * 100).toFixed(1);

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <div className="mb-4">
        <h2 className="text-xl font-semibold text-slate-900">
          Feature contributions (SHAP)
        </h2>
        <p className="text-sm text-slate-500 mt-1">
          Baseline {baseline}% → predicted {predicted}%. Each bar shows one
          feature's effect on this patient's probability.
        </p>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 8, right: 24, bottom: 8, left: 80 }}
        >
          <XAxis type="number" tickFormatter={(v) => v.toFixed(2)} />
          <YAxis type="category" dataKey="feature" width={120} />
          <ReferenceLine x={0} stroke="#94a3b8" />
          <Tooltip
            formatter={(value: number) => value.toFixed(3)}
            labelFormatter={(label: string) => `Feature: ${label}`}
          />
          <Bar dataKey="shap_value">
            {data.map((entry, idx) => (
              <Cell
                key={idx}
                fill={entry.shap_value >= 0 ? "#dc2626" : "#16a34a"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <p className="text-xs text-slate-500 mt-3">
        Red bars increased readmission risk for this patient. Green bars
        decreased it. Bars sum to the difference between baseline and
        predicted probability.
      </p>
    </div>
  );
}
