"use client";

import { useMemo, useState } from "react";
import { api, labels, type ValidationItem, type ValidationResult } from "@/lib/types";

const PAGE = 24;
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

interface GarmentSummary { label: string; total: number; correct: number; topMistake: { label: string; count: number } | null }

function summarize(items: ValidationItem[]): GarmentSummary[] {
  return labels.map(label => {
    const rows = items.filter(i => i.label === label);
    const wrong = new Map<string, number>();
    for (const row of rows) if (!row.correct) wrong.set(row.predicted, (wrong.get(row.predicted) ?? 0) + 1);
    const [top] = [...wrong.entries()].sort((a, b) => b[1] - a[1]);
    return { label, total: rows.length, correct: rows.length - [...wrong.values()].reduce((a, b) => a + b, 0),
      topMistake: top ? { label: top[0], count: top[1] } : null };
  });
}

export function ValidationPanel({ modelId }: { modelId: string }) {
  const [result, setResult] = useState<ValidationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [show, setShow] = useState<"mistakes" | "all">("mistakes");
  const [garment, setGarment] = useState<string | null>(null);
  const [limit, setLimit] = useState(PAGE);

  async function load() {
    if (result || loading) return;
    setLoading(true); setError(null);
    try { setResult(await api<ValidationResult>(`/models/${modelId}/validation`)); }
    catch (e) { setError((e as Error).message); }
    finally { setLoading(false); }
  }
  function filter(next: { show?: "mistakes" | "all"; garment?: string | null }) {
    if (next.show) setShow(next.show);
    if (next.garment !== undefined) setGarment(next.garment);
    setLimit(PAGE);
  }

  const summary = useMemo(() => summarize(result?.items ?? []), [result]);
  const inGarment = (result?.items ?? []).filter(i => !garment || i.label === garment);
  const mistakeCount = inGarment.filter(i => !i.correct).length;
  // Mistakes first, then the least certain predictions.
  const items = inGarment.filter(i => show === "all" || !i.correct)
    .sort((a, b) => Number(a.correct) - Number(b.correct) || a.score - b.score);
  const visible = items.slice(0, limit);

  return <details className="validation-panel" onToggle={e => { if (e.currentTarget.open) void load(); }}>
    <summary>View validation images</summary>
    <p className="muted">View-only. These are the images this model was scored on in its training run. They are never used for training and can’t be relabelled, so every model is compared on equal terms. Reserved test images stay hidden.</p>
    {loading && <p role="status">Loading predictions. Transformer models can take a moment the first time.</p>}
    {error && <p className="run-error" role="alert">{error} <button className="text-button" onClick={() => void load()}>Try again</button></p>}
    {result && <>
      <p className="validation-total"><strong>{result.correct} of {result.samples}</strong> validation images predicted correctly. Choose a garment to see its images.</p>

      <div className="garment-breakdown" role="group" aria-label="Results by actual garment">
        {summary.map(row => {
          const selected = garment === row.label;
          return <button key={row.label} className={`garment-row ${selected ? "selected" : ""}`} aria-pressed={selected}
            onClick={() => filter({ garment: selected ? null : row.label })}>
            <span className="garment-name">{row.label}</span>
            <span className="garment-bar" aria-hidden="true"><span style={{ width: row.total ? `${(row.correct / row.total) * 100}%` : 0 }} /></span>
            <span className="garment-count">{row.correct} of {row.total}</span>
            <span className="garment-mixup">{row.topMistake ? `Mostly mistaken for ${row.topMistake.label} (${row.topMistake.count})` : row.total ? "No mistakes" : "No images"}</span>
          </button>;
        })}
      </div>

      <div className="validation-toolbar">
        <div className="segmented" role="tablist" aria-label="Which images to show">
          <button role="tab" aria-selected={show === "mistakes"} className={show === "mistakes" ? "active" : ""} onClick={() => filter({ show: "mistakes" })}>Mistakes <span>{mistakeCount}</span></button>
          <button role="tab" aria-selected={show === "all"} className={show === "all" ? "active" : ""} onClick={() => filter({ show: "all" })}>All <span>{inGarment.length}</span></button>
        </div>
        <span className="muted">{garment ? <>Actual garment: <strong>{garment}</strong> <button className="text-button" onClick={() => filter({ garment: null })}>Show all garments</button></> : "All garments"}</span>
      </div>

      {visible.length ? <div className="validation-grid">{visible.map(item => <figure className={`validation-item ${item.correct ? "correct" : "mistake"}`} key={item.image_id}>
        <img src={`/api/images/${item.image_id}/file`} alt={`Validation image of a ${item.label}`} loading="lazy" />
        <figcaption>
          <span className="visually-hidden">{item.correct ? "Correct." : "Mistake."}</span>
          <span>Actual <strong>{item.label}</strong></span>
          <span>Predicted <strong>{item.predicted}</strong> <em>{percent(item.score)}</em></span>
        </figcaption>
      </figure>)}</div> : <p className="muted">{show === "mistakes" ? "No mistakes here. Switch to All to see these images." : "No validation images for this garment."}</p>}

      {items.length > limit && <button className="secondary show-more" onClick={() => setLimit(limit + PAGE)}>
        Show {Math.min(PAGE, items.length - limit)} more ({items.length - limit} left)
      </button>}
      <p className="muted">Showing {visible.length} of {items.length}. Sorted with mistakes first, then the least certain predictions. Scores are not calibrated confidence.</p>
    </>}
  </details>;
}
