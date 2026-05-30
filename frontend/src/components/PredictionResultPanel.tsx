/**
 * PredictionResultPanel — clinical-style risk readout.
 *
 * Renders the readmission probability as a large percentage with a
 * color-coded risk band badge underneath, plus a short interpretation line.
 */
import type { PredictionResponse, RiskBand } from "../types/api";


interface Props {
  prediction: PredictionResponse;
}


const BAND_STYLES: Record<RiskBand, { bg: string; text: string; label: string }> = {
  low: {
    bg: "bg-emerald-100",
    text: "text-emerald-800",
    label: "Low risk",
  },
  medium: {
    bg: "bg-amber-100",
    text: "text-amber-800",
    label: "Medium risk",
  },
  high: {
    bg: "bg-red-100",
    text: "text-red-800",
    label: "High risk",
  },
};


const BAND_DESCRIPTION: Record<RiskBand, string> = {
  low: "Below the threshold for proactive intervention.",
  medium: "Worth flagging for clinical review before discharge.",
  high: "Strongly consider extended care, follow-up scheduling, or case management.",
};


export function PredictionResultPanel({ prediction }: Props) {
  const { mrn, readmission_probability, risk_band } = prediction;
  const style = BAND_STYLES[risk_band];
  const percentage = (readmission_probability * 100).toFixed(1);

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-semibold text-slate-900">
            Prediction
          </h2>
          <p className="text-sm text-slate-500">Patient: {mrn}</p>
        </div>
        <span
          className={`px-3 py-1 rounded-full text-sm font-medium ${style.bg} ${style.text}`}
        >
          {style.label}
        </span>
      </div>

      <div className="flex items-baseline gap-3">
        <span className="text-5xl font-bold text-slate-900">{percentage}%</span>
        <span className="text-slate-500">30-day readmission probability</span>
      </div>

      <p className="text-sm text-slate-600 mt-4">
        {BAND_DESCRIPTION[risk_band]}
      </p>
    </div>
  );
}
