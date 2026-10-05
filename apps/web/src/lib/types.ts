export const labels = ["Dress", "T-shirt", "Shirt", "Sweater", "Blouse"] as const;
export type Label = (typeof labels)[number];
export type View = "classify" | "review" | "training" | "models";
export type Trainer = "baseline" | "vit";
export interface Metrics {
  accuracy: number; macro_f1: number; samples: number; training_samples?: number;
  confusion_matrix: number[][]; labels: string[]; gate_passed?: boolean;
  gate_threshold?: number; algorithm?: string;
  per_class?: { label: string; precision: number; recall: number; f1: number; samples: number }[];
}
export interface Overview {
  class_splits?: Record<string, { train: number; validation: number; test: number }>;
  labels: string[]; dataset: string; images: number; reviewed: number; pending: number;
  class_counts: Record<string, number>; split_counts: Record<string, number>; runs: number;
  active_model: { id: string; metrics: Metrics } | null; worker_heartbeat: string | null;
}
export interface Garment {
  id: string; filename: string; source: string; label: string | null; split: string;
  reviewed_label?: string | null; predicted_label?: string | null; model_id?: string | null;
}
export interface Prediction { label: string; model_id: string; scores: { label: string; score: number }[] }
export interface Classification { image: Garment; prediction: Prediction | null }
export interface TrainingRun {
  trainer: Trainer;
  id: string; status: "queued" | "running" | "completed" | "failed"; step: string;
  progress: number; error: string | null; metrics: Metrics | null; baseline_metrics: Metrics | null;
  created_at: string; finished_at: string | null;
}
export interface ModelVersion {
  id: string; run_id: string; metrics: Metrics; eligible: boolean; created_at: string;
  active: boolean; previously_active: boolean; can_activate: boolean;
  test_report?: TestReport | null;
}
export interface TestReport {
  id: string; model_id: string; status: "queued" | "running" | "completed" | "failed";
  metrics: Metrics | null; error: string | null; created_at: string; finished_at: string | null;
}

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(`/api${path}`, options); }
  catch { throw new Error("The local service is unavailable. Start VisionLoop with npm run dev."); }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === "string" ? body.detail : `Request failed (${response.status}). Check that the local API is running.`);
  }
  return response.json() as Promise<T>;
}
