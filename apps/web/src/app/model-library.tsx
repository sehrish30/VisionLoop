"use client";

import { useState } from "react";
import { Check, RotateCcw, Sparkles, Trash2 } from "lucide-react";
import type { ModelVersion } from "@/lib/types";
import { TestReportPanel } from "./test-report-panel";
import { ValidationPanel } from "./validation-panel";

const RECENT_CANDIDATES = 3;
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

interface Actions {
  busy: string | null;
  jobRunning: boolean;
  onActivate: (id: string) => void;
  onRequestReport: (id: string) => void;
  onDelete: (model: ModelVersion) => void;
}

export function ModelLibrary({ models, onCleanup, ...actions }: Actions & {
  models: ModelVersion[]; onCleanup: (count: number) => void;
}) {
  const [showOlder, setShowOlder] = useState(false);
  const active = models.filter(m => m.active);
  const earlier = models.filter(m => !m.active && m.previously_active);
  const candidates = models.filter(m => !m.active && !m.previously_active);
  const shownCandidates = showOlder ? candidates : candidates.slice(0, RECENT_CANDIDATES);
  const hidden = candidates.length - shownCandidates.length;
  const failed = candidates.filter(m => !m.eligible).length;

  return <div className="model-library">
    <div className="library-toolbar">
      <p className="muted">{models.length} {models.length === 1 ? "model" : "models"}. Training runs and their scores stay in Training runs even after a model is deleted.</p>
      {failed > 0 && <button className="secondary" disabled={!!actions.busy || actions.jobRunning} title={actions.jobRunning ? "Wait for the current job to finish" : undefined} onClick={() => onCleanup(failed)}>
        <Trash2 size={16} />Delete candidates that didn’t pass ({failed})
      </button>}
    </div>

    {active.length > 0 && <section className="library-group" aria-labelledby="group-active">
      <h3 id="group-active">Active model</h3>
      <div className="model-list">{active.map(m => <ModelCard key={m.id} model={m} {...actions} />)}</div>
    </section>}

    {earlier.length > 0 && <section className="library-group" aria-labelledby="group-earlier">
      <h3 id="group-earlier">Earlier active versions <span>{earlier.length}</span></h3>
      <p className="muted">You can restore any of these.</p>
      <div className="model-list">{earlier.map(m => <ModelCard key={m.id} model={m} {...actions} />)}</div>
    </section>}

    {candidates.length > 0 && <section className="library-group" aria-labelledby="group-candidates">
      <h3 id="group-candidates">Candidates <span>{candidates.length}</span></h3>
      <p className="muted">Models from training runs that have never been active, newest first.</p>
      <div className="model-list">{shownCandidates.map(m => <ModelCard key={m.id} model={m} {...actions} />)}</div>
      {hidden > 0 && <button className="secondary show-more" onClick={() => setShowOlder(true)}>Show {hidden} older {hidden === 1 ? "candidate" : "candidates"}</button>}
      {showOlder && candidates.length > RECENT_CANDIDATES && <button className="text-button show-more" onClick={() => setShowOlder(false)}>Show only the {RECENT_CANDIDATES} newest</button>}
    </section>}
  </div>;
}

function ModelCard({ model, busy, jobRunning, onActivate, onRequestReport, onDelete }: Actions & { model: ModelVersion }) {
  return <article className={`model-card ${model.active ? "is-active" : ""}`}>
    <div className="run-header">
      <div>
        <div className="model-name"><h2>{model.id}</h2>{model.active && <span className="status-tag completed">Active</span>}</div>
        <p>{model.metrics.algorithm}</p>
      </div>
      <div className="model-actions">
        <button className={model.active ? "secondary" : "primary"} disabled={model.active || !model.can_activate || !!busy} onClick={() => onActivate(model.id)}>
          {model.active ? <Check size={16} /> : model.previously_active ? <RotateCcw size={16} /> : <Sparkles size={16} />}
          {model.active ? "In use" : model.previously_active ? "Restore version" : "Activate model"}
        </button>
        {!model.active && <button className="icon-button danger" aria-label={`Delete model ${model.id}`}
          title={jobRunning ? "Wait for the current job to finish" : "Delete this model"}
          disabled={!!busy || jobRunning} onClick={() => onDelete(model)}><Trash2 size={17} /></button>}
      </div>
    </div>
    <div className="model-metrics">
      <div><span>Validation accuracy</span><strong>{percent(model.metrics.accuracy)}</strong></div>
      <div><span>Validation macro F1</span><strong>{percent(model.metrics.macro_f1)}</strong></div>
      <div><span>Training images</span><strong>{model.metrics.training_samples}</strong></div>
      <div><span>Quality gate</span><strong className="gate-value">{model.eligible ? "Passed" : "Not passed"}</strong></div>
    </div>
    <details><summary>Inspect validation confusion matrix</summary>
      <p className="muted">Rows are actual labels; columns are predictions. Larger diagonal values are better.</p>
      <div className="table-scroll"><table><thead><tr><th>Actual / predicted</th>{model.metrics.labels.map(l => <th key={l}>{l}</th>)}</tr></thead>
        <tbody>{model.metrics.confusion_matrix.map((row, i) => <tr key={i}><th>{model.metrics.labels[i]}</th>{row.map((n, j) => <td className={i === j ? "diagonal" : ""} key={j}>{n}</td>)}</tr>)}</tbody></table></div>
    </details>
    <ValidationPanel modelId={model.id} />
    <TestReportPanel model={model} busy={!!busy} onRequest={onRequestReport} />
  </article>;
}
