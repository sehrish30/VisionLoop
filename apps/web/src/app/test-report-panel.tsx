"use client";

import type { ModelVersion } from "@/lib/types";

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

export function TestReportPanel({ model, busy, onRequest }: {
  model: ModelVersion; busy: boolean; onRequest: (id: string) => void;
}) {
  const report = model.test_report;
  if (!model.active && !report) return null;
  const pending = report?.status === "queued" || report?.status === "running";
  const metrics = report?.metrics;
  return <section className="final-test" aria-label={`Final test report for ${model.id}`}>
    <h3>Final test report</h3>
    <p className="muted">Use this after choosing your model with validation results. It evaluates reserved test images from this model’s training snapshot without training or switching models.</p>
    <p className="muted">Using test results to choose or tune future models makes this test set part of development. You would then need a fresh held-out set for a final assessment.</p>
    {pending && <p role="status">{report.status === "queued" ? "Queued — waiting for the local worker." : "Evaluating reserved test images…"}</p>}
    {report?.error && <p className="run-error" role="alert">{report.error}</p>}
    {model.active && (!report || report.status === "failed") && <button className="secondary" disabled={busy} onClick={() => onRequest(model.id)}>{report ? "Retry test report" : "Evaluate chosen model on test images"}</button>}
    {report?.status === "completed" && metrics && <>
      <p className="muted">Saved {new Date(report.finished_at ?? report.created_at).toLocaleString()}. This result is fixed; later imports do not change it.</p>
      <div className="run-metrics"><span>Test accuracy <strong>{percent(metrics.accuracy)}</strong></span><span>Test macro F1 <strong>{percent(metrics.macro_f1)}</strong></span><span>Test images <strong>{metrics.samples}</strong></span></div>
      <details><summary>Inspect test results by class</summary>
        <p className="muted">Precision: how often a predicted label was right. Recall: how many actual examples of that class were found. F1 balances both.</p>
        <div className="table-scroll"><table><thead><tr><th scope="col">Garment</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">F1</th><th scope="col">Images</th></tr></thead>
          <tbody>{metrics.per_class?.map(row => <tr key={row.label}><th scope="row">{row.label}</th><td>{percent(row.precision)}</td><td>{percent(row.recall)}</td><td>{percent(row.f1)}</td><td>{row.samples}</td></tr>)}</tbody></table></div>
        <p className="muted">Confusion matrix: rows are actual labels; columns are predictions.</p>
        <div className="table-scroll"><table><thead><tr><th scope="col">Actual / predicted</th>{metrics.labels.map(label => <th scope="col" key={label}>{label}</th>)}</tr></thead>
          <tbody>{metrics.confusion_matrix.map((row, i) => <tr key={metrics.labels[i]}><th scope="row">{metrics.labels[i]}</th>{row.map((count, j) => <td className={i === j ? "diagonal" : ""} key={j}>{count}</td>)}</tr>)}</tbody></table></div>
      </details>
      <a className="text-button" href={`/api/models/${model.id}/test-report`} download>Download test report (JSON)</a>
    </>}
  </section>;
}
