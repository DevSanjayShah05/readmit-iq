/**
 * Typed API client for the ReadmitIQ backend.
 *
 * Wraps fetch() so components don't deal with the browser HTTP API directly.
 * Returns typed responses; throws on non-2xx status.
 */
import type {
  BatchExplanationResponse,
  BatchPredictionResponse,
  BatchPredictRequest,
  PatientRequest,
} from "../types/api";


export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}


async function postJson<TRequest, TResponse>(
  path: string,
  body: TRequest,
): Promise<TResponse> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new ApiError(
      response.status,
      `API error ${response.status}: ${detail}`,
    );
  }

  return response.json() as Promise<TResponse>;
}


/** Score one or more patients. */
export async function predict(
  patients: PatientRequest[],
): Promise<BatchPredictionResponse> {
  const body: BatchPredictRequest = { patients };
  return postJson<BatchPredictRequest, BatchPredictionResponse>(
    "/api/predict",
    body,
  );
}


/** Generate SHAP explanations for one or more patients. */
export async function explain(
  patients: PatientRequest[],
): Promise<BatchExplanationResponse> {
  const body: BatchPredictRequest = { patients };
  return postJson<BatchPredictRequest, BatchExplanationResponse>(
    "/api/explain",
    body,
  );
}
