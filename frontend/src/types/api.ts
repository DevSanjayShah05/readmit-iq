/**
 * TypeScript mirrors of the ReadmitIQ backend Pydantic schemas.
 *
 * Source of truth: src/readmit_iq/api/schemas.py
 *
 * Keep these in sync with the backend by hand. If a field changes shape,
 * change it here too. The compiler will catch missing or misnamed fields
 * across all frontend code that uses these types.
 */

// ---------- Request shapes ----------

export interface PatientRequest {
  mrn: string;
  age: number;
  sex: "F" | "M" | "O";
  admission_date: string;  // ISO date string, e.g. "2024-06-15"
  discharge_date: string;
  primary_diagnosis: string | null;
}

export interface BatchPredictRequest {
  patients: PatientRequest[];
}

// ---------- Response shapes ----------

export type RiskBand = "low" | "medium" | "high";

export interface PredictionResponse {
  mrn: string;
  readmission_probability: number;
  risk_band: RiskBand;
}

export interface BatchPredictionResponse {
  predictions: PredictionResponse[];
}

export interface FeatureContributionResponse {
  feature_name: string;
  feature_value: number;
  shap_value: number;
}

export interface ExplanationResponse {
  mrn: string;
  predicted_probability: number;
  baseline_probability: number;
  contributions: FeatureContributionResponse[];
}

export interface BatchExplanationResponse {
  explanations: ExplanationResponse[];
}
