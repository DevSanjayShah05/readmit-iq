/**
 * PatientForm — input form for predicting readmission risk on one patient.
 *
 * Holds the form values in local state (controlled inputs), submits to the
 * /predict endpoint via the API client, and renders the response (or an
 * error) below the form.
 */
import { useState } from "react";

import { ApiError, predict } from "../api/client";
import type {
  BatchPredictionResponse,
  PatientRequest,
} from "../types/api";


type Status = "idle" | "loading" | "success" | "error";


/** Default values shown in the form on first render. */
const DEFAULT_VALUES: PatientRequest = {
  mrn: "DEMO-001",
  age: 78,
  sex: "M",
  admission_date: "2024-06-15",
  discharge_date: "2024-06-22",
  primary_diagnosis: "I50.9",
};


export function PatientForm() {
  // Form values (controlled inputs)
  const [values, setValues] = useState<PatientRequest>(DEFAULT_VALUES);

  // Request lifecycle state
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<BatchPredictionResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Generic field updater: builds a new values object with one field changed.
  // Using a callback ensures we read the latest state if React batches updates.
  function updateField<K extends keyof PatientRequest>(
    field: K,
    value: PatientRequest[K],
  ): void {
    setValues((prev) => ({ ...prev, [field]: value }));
  }

  async function handleSubmit(event: React.FormEvent): Promise<void> {
    event.preventDefault(); // don't reload the page on form submit
    setStatus("loading");
    setErrorMessage(null);
    setResult(null);

    try {
      const response = await predict([values]);
      setResult(response);
      setStatus("success");
    } catch (err) {
      if (err instanceof ApiError) {
        setErrorMessage(err.message);
      } else if (err instanceof Error) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage("Unknown error");
      }
      setStatus("error");
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="bg-white p-6 rounded-lg shadow-md space-y-4"
    >
      <h2 className="text-xl font-semibold text-slate-900">Patient details</h2>

      <div className="grid grid-cols-2 gap-4">
        <label className="block">
          <span className="text-sm font-medium text-slate-700">MRN</span>
          <input
            type="text"
            value={values.mrn}
            onChange={(e) => updateField("mrn", e.target.value)}
            required
            maxLength={64}
            className="mt-1 block w-full rounded border-slate-300 shadow-sm"
          />
        </label>

        <label className="block">
          <span className="text-sm font-medium text-slate-700">Age</span>
          <input
            type="number"
            value={values.age}
            onChange={(e) => updateField("age", Number(e.target.value))}
            required
            min={0}
            max={120}
            className="mt-1 block w-full rounded border-slate-300 shadow-sm"
          />
        </label>

        <label className="block">
          <span className="text-sm font-medium text-slate-700">Sex</span>
          <select
            value={values.sex}
            onChange={(e) =>
              updateField("sex", e.target.value as PatientRequest["sex"])
            }
            className="mt-1 block w-full rounded border-slate-300 shadow-sm"
          >
            <option value="F">F</option>
            <option value="M">M</option>
            <option value="O">O</option>
          </select>
        </label>

        <label className="block">
          <span className="text-sm font-medium text-slate-700">
            Primary diagnosis (ICD-10)
          </span>
          <input
            type="text"
            value={values.primary_diagnosis ?? ""}
            onChange={(e) =>
              updateField("primary_diagnosis", e.target.value || null)
            }
            maxLength={10}
            className="mt-1 block w-full rounded border-slate-300 shadow-sm"
          />
        </label>

        <label className="block">
          <span className="text-sm font-medium text-slate-700">
            Admission date
          </span>
          <input
            type="date"
            value={values.admission_date}
            onChange={(e) => updateField("admission_date", e.target.value)}
            required
            className="mt-1 block w-full rounded border-slate-300 shadow-sm"
          />
        </label>

        <label className="block">
          <span className="text-sm font-medium text-slate-700">
            Discharge date
          </span>
          <input
            type="date"
            value={values.discharge_date}
            onChange={(e) => updateField("discharge_date", e.target.value)}
            required
            className="mt-1 block w-full rounded border-slate-300 shadow-sm"
          />
        </label>
      </div>

      <button
        type="submit"
        disabled={status === "loading"}
        className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-medium py-2 px-4 rounded transition"
      >
        {status === "loading" ? "Predicting…" : "Predict"}
      </button>

      {status === "error" && errorMessage && (
        <div className="p-3 rounded bg-red-50 border border-red-200 text-red-800 text-sm">
          {errorMessage}
        </div>
      )}

      {status === "success" && result && (
        <div className="p-3 rounded bg-slate-50 border border-slate-200">
          <h3 className="text-sm font-semibold text-slate-700 mb-2">
            Response
          </h3>
          <pre className="text-xs text-slate-800 overflow-x-auto">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </form>
  );
}
