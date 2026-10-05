"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { TestReportPanel } from "./test-report-panel";
import { ArrowUpRight, Check, CheckCheck, ChevronRight, CircleHelp, Database, FlaskConical, ImagePlus, Layers3, LoaderCircle, RotateCcw, ScanLine, Shirt, Sparkles, Upload, X } from "lucide-react";
import { api, labels, type Classification, type Garment, type ModelVersion, type Overview, type Trainer, type TrainingRun, type View } from "@/lib/types";

const navigation = [
  { id: "classify" as View, title: "Classify", icon: ScanLine },
  { id: "review" as View, title: "Review queue", icon: CheckCheck },
  { id: "training" as View, title: "Training runs", icon: FlaskConical },
  { id: "models" as View, title: "Model library", icon: Layers3 },
];
const titles: Record<View, [string, string]> = {
  classify: ["A second look. A smarter label.", "Turn a clothing photo into a garment label. Every correction helps the next model learn."],
  review: ["Your eye makes the difference.", "Review uploaded pieces and give the next training run better labels."],
  training: ["Make the next model better.", "Follow each local run from a fixed dataset snapshot to an evaluated candidate."],
  models: ["A history of learning.", "Compare candidates, choose your active model, and return to an earlier version."],
};
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const date = (value: string) => new Date(value).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

export default function Studio() {
  const [view, setView] = useState<View>("classify");
  const [overview, setOverview] = useState<Overview | null>(null);
  const [images, setImages] = useState<Garment[]>([]);
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [trainer, setTrainer] = useState<Trainer>("baseline");
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [result, setResult] = useState<Classification | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [reviewFilter, setReviewFilter] = useState<"pending" | "all">("pending");
  const [reviewLabels, setReviewLabels] = useState<Record<string, string>>({});
  const [setupOpen, setSetupOpen] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [o, i, r, m] = await Promise.all([api<Overview>("/overview"), api<Garment[]>("/images"), api<TrainingRun[]>("/runs"), api<ModelVersion[]>("/models")]);
      setOverview(o); setImages(i); setRuns(r); setModels(m); setConnected(true);
    } catch { setConnected(false); }
  }, []);
  useEffect(() => { void refresh(); const timer = setInterval(() => void refresh(), 4000); return () => clearInterval(timer); }, [refresh]);
  useEffect(() => {
    if (!selectedFile) { setPreview(null); return; }
    const url = URL.createObjectURL(selectedFile); setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(null), 5000);
    return () => clearTimeout(timer);
  }, [notice]);

  function choose(file?: File) {
    if (!file) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || file.size > 8 * 1024 * 1024) {
      setError("Choose a JPG, PNG, or WebP image under 8 MB."); return;
    }
    setSelectedFile(file); setResult(null); setError(null);
  }
  async function upload() {
    if (!selectedFile) return;
    setBusy("upload"); setError(null);
    try {
      const form = new FormData(); form.append("file", selectedFile);
      setResult(await api<Classification>("/classify", { method: "POST", body: form }));
      await refresh();
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(null); }
  }
  async function saveReview(image: Garment, label: string) {
    setBusy(image.id); setError(null);
    try {
      await api(`/images/${image.id}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ label }) });
      setNotice(`Saved as ${label}. Ready for the next training snapshot.`); await refresh();
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(null); }
  }
  async function train() {
    setBusy("train"); setError(null);
    try { await api("/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ trainer }) }); setNotice("Training queued. The local worker will pick it up."); await refresh(); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(null); }
  }
  async function activate(id: string) {
    setBusy(id); setError(null);
    try { await api(`/models/${id}/activate`, { method: "POST" }); setNotice("Active model updated. New predictions will use this version."); await refresh(); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(null); }
  }
  async function requestTestReport(id: string) {
    setBusy(id); setError(null);
    try {
      await api(`/models/${id}/test-report`, { method: "POST" });
      setNotice("Final test report queued. The local worker will evaluate the saved model.");
      await refresh();
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(null); }
  }

  const queued = runs.some(r => r.status === "running" || r.status === "queued");
  const shownImages = images.filter(i => i.split === "train" && (reviewFilter === "all" || !i.reviewed_label));
  const workerReady = overview?.worker_heartbeat && Date.now() - new Date(overview.worker_heartbeat).getTime() < 20000;
  const imageUrl = result ? `/api/images/${result.image.id}/file` : preview;

  return <div className="studio">
    <aside className="sidebar">
      <a href="/" className="brand" aria-label="VisionLoop home"><span className="brand-symbol"><ScanLine size={24} /></span><span>VisionLoop<span className="brand-sub">Clothing intelligence</span></span></a>
      <div className="workspace-label">Your workspace</div>
      <nav aria-label="Main navigation">{navigation.map(({ id, title, icon: Icon }) => <button key={id} onClick={() => { setView(id); setError(null); }} className={`nav-item ${view === id ? "selected" : ""}`} aria-current={view === id ? "page" : undefined}><Icon size={19} /><span>{title}</span>{id === "review" && !!overview?.pending && <span className="count">{overview.pending}</span>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="local-note"><span className={`status-dot ${connected ? "online" : ""}`} /><div>{connected ? "Running locally" : "Connecting to service"}<small>No cloud jobs</small></div></div><button className="help-button" onClick={() => setSetupOpen(true)}><CircleHelp size={17} /> Getting started <ArrowUpRight size={15} /></button><div className="version">VisionLoop / Learning edition</div></div>
    </aside>
    <div className="main-shell">
      <header className="topbar"><div className="breadcrumb">Workspace <ChevronRight size={14} /><strong>{navigation.find(n => n.id === view)?.title}</strong></div><span className="local-pill"><span className="status-dot online" /> Local workspace</span></header>
      <main>
        <div className="page-heading"><div><h1>{titles[view][0]}</h1><p>{titles[view][1]}</p></div>{view === "training" && <button className="primary" onClick={() => void train()} disabled={!!busy || queued || !connected}><FlaskConical size={17} />{queued ? "Run in progress" : "Start training"}</button>}</div>
        {!connected && <div className="connection-banner">The local API is starting or unavailable. <button onClick={() => setSetupOpen(true)}>View setup instructions</button></div>}
        {error && <div className="message error" role="alert">{error}<button aria-label="Dismiss error" onClick={() => setError(null)}><X size={17} /></button></div>}
        {notice && <div className="message success" role="status"><Check size={17} />{notice}<button aria-label="Dismiss notification" onClick={() => setNotice(null)}><X size={17} /></button></div>}

        {view === "classify" && <>
          <div className="classify-grid">
            <section className="image-panel">
              <div className="panel-heading"><h2>Garment photo</h2><span>One piece at a time</span></div>
              <div className={`upload-zone ${dragging ? "dragging" : ""} ${imageUrl ? "has-image" : ""}`} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); if (!busy) choose(e.dataTransfer.files[0]); }}>
                {imageUrl ? <><img className="garment-preview" src={imageUrl} alt={selectedFile?.name ?? "Uploaded garment"} /><button className="change-photo" disabled={!!busy} onClick={() => fileInput.current?.click()}><RotateCcw size={15} />Change photo</button></> : <div className="upload-content"><div className="garment-outline"><Shirt size={82} strokeWidth={0.9} /><span className="scan-corner top-left" /><span className="scan-corner top-right" /><span className="scan-corner bottom-left" /><span className="scan-corner bottom-right" /></div><h3>A new perspective on pre-loved.</h3><p>Drop a clothing photo here,<br />or choose one from your computer.</p><button className="primary" onClick={() => fileInput.current?.click()}><Upload size={17} /> Choose image</button><small>JPG, PNG or WebP · Up to 8 MB</small></div>}
              </div>
              <input ref={fileInput} className="visually-hidden" type="file" aria-label="Choose garment image" accept="image/jpeg,image/png,image/webp" onChange={e => { choose(e.target.files?.[0]); e.target.value = ""; }} />
              <div className="photo-footer"><span><ScanLine size={16} /> Front view. Clear background. Best results.</span>{selectedFile && <button className="primary small" onClick={() => void upload()} disabled={!!busy || !connected}>{busy === "upload" ? <LoaderCircle size={15} className="spin" /> : <Sparkles size={15} />}{overview?.active_model ? "Classify image" : "Save for review"}</button>}</div>
            </section>
            <section className="prediction-panel"><div className="panel-heading"><h2>Prediction</h2><span className="subtle-badge">{result?.prediction ? "Ready" : "Awaiting image"}</span></div>
              {result?.prediction ? <div className="prediction-result"><div className="prediction-icon"><Shirt size={27} /></div><p className="muted">Suggested garment type</p><h3>{result.prediction.label}</h3><div className="scores">{result.prediction.scores.map(s => <div className="score" key={s.label}><div><span>{s.label}</span><strong>{percent(s.score)}</strong></div><div className="score-track"><div style={{ width: percent(s.score) }} /></div></div>)}</div><small>Model scores are not calibrated confidence.</small></div> : <div className="prediction-empty"><div className="prediction-icon"><Layers3 size={28} strokeWidth={1.4} /></div><h3>{result ? "Saved to your workspace" : overview?.active_model ? "Ready when you are" : "Your model starts here"}</h3><p>{result ? "Add the correct label below. Your image will be available for the next training run." : overview?.active_model ? "Upload an image to see its suggested type and model scores." : "Import a small dataset, train a model, then activate it to start making predictions."}</p>{!result && !overview?.active_model && <button className="text-button" onClick={() => setSetupOpen(true)}>Set up your first model <ChevronRight size={15} /></button>}</div>}
              {result && result.image.split === "train" && <div className="review-inline"><label htmlFor="correct-label">What is the correct garment type?</label><div><select id="correct-label" value={reviewLabels[result.image.id] ?? result.prediction?.label ?? labels[0]} onChange={e => setReviewLabels({ ...reviewLabels, [result.image.id]: e.target.value })}>{labels.map(label => <option key={label}>{label}</option>)}</select><button className="primary small" disabled={!!busy} onClick={() => void saveReview(result.image, reviewLabels[result.image.id] ?? result.prediction?.label ?? labels[0])}><Check size={16} />Save</button></div></div>}
              <div className="model-footer"><span className={`status-dot ${overview?.active_model ? "online" : ""}`} /><span>{overview?.active_model ? `Active: ${overview.active_model.id}` : "No active model yet"}</span></div>
            </section>
          </div>
          <section className="dataset-strip"><div className="dataset-icon"><Database size={24} strokeWidth={1.5} /></div><div className="dataset-description"><h2>Made for second-hand fashion</h2><p>Five garment types. Real clothing photos. A little better with every review.</p><a href="https://huggingface.co/datasets/fnauman/fashion-second-hand-front-only-rgb" target="_blank" rel="noreferrer">Explore the source dataset <ArrowUpRight size={13} /></a></div><div className="label-chips">{labels.map(label => <span key={label}>{label}</span>)}</div></section>
          <div className="workspace-summary"><span><strong>{overview?.images ?? 0}</strong> local images</span><span><strong>{overview?.reviewed ?? 0}</strong> labeled pieces</span><span><strong>{overview?.runs ?? 0}</strong> training runs</span><button onClick={() => setView("training")}>Explore your training loop <ChevronRight size={15} /></button></div>
        </>}

        {view === "review" && <>
          <div className="section-toolbar"><div className="segmented"><button className={reviewFilter === "pending" ? "active" : ""} onClick={() => setReviewFilter("pending")}>Needs a label <span>{overview?.pending ?? 0}</span></button><button className={reviewFilter === "all" ? "active" : ""} onClick={() => setReviewFilter("all")}>All training images</button></div><span className="muted">Evaluation images are kept separate.</span></div>
          {shownImages.length ? <div className="review-grid">{shownImages.map(image => <article className="review-card" key={image.id}><div className="review-photo"><img src={`/api/images/${image.id}/file`} alt={image.filename} loading="lazy" /></div><div className="review-card-body"><div className="review-card-title"><strong>{image.reviewed_label ?? "Unlabeled garment"}</strong>{image.reviewed_label && <CheckCheck size={17} />}</div><p>{image.predicted_label ? `Predicted: ${image.predicted_label}` : image.source.startsWith("huggingface:") ? "Dataset label" : "Uploaded photo"}</p><label className="visually-hidden" htmlFor={`label-${image.id}`}>Correct label for {image.filename}</label><div className="label-control"><select id={`label-${image.id}`} value={reviewLabels[image.id] ?? image.reviewed_label ?? image.predicted_label ?? labels[0]} onChange={e => setReviewLabels({ ...reviewLabels, [image.id]: e.target.value })}>{labels.map(l => <option key={l}>{l}</option>)}</select><button className="icon-button" aria-label={`Save label for ${image.filename}`} disabled={!!busy} onClick={() => void saveReview(image, reviewLabels[image.id] ?? image.reviewed_label ?? image.predicted_label ?? labels[0])}><Check size={18} /></button></div></div></article>)}</div> : <Empty icon={<CheckCheck size={34} />} title="A clear review queue" text="Upload a clothing photo to start collecting labels for your next model." action="Classify a photo" onAction={() => setView("classify")} />}
        </>}

        {view === "training" && <>
          <div className="training-choice">
            <h2>Dataset coverage</h2>
            <p>Training images teach the model. Validation images compare versions. Reserved test images stay out of training and model selection.</p>
            {overview?.class_splits ? <div className="table-scroll"><table>
              <thead><tr><th scope="col">Garment</th><th scope="col">Training</th><th scope="col">Validation</th><th scope="col">Reserved test</th></tr></thead>
              <tbody>{labels.map(label => <tr key={label}><th scope="row">{label}</th><td>{overview.class_splits?.[label]?.train ?? 0}</td><td>{overview.class_splits?.[label]?.validation ?? 0}</td><td>{overview.class_splits?.[label]?.test ?? 0}</td></tr>)}</tbody>
            </table></div> : <p>Coverage is unavailable. Restart the local API if it is already running.</p>}
            {overview?.class_splits && labels.some(label => (overview.class_splits?.[label]?.validation ?? 0) < 20) && <p>Some classes have fewer than 20 validation images. Scores may change sharply with just a few predictions. A larger sample helps, but this is still a learning experiment.</p>}
            <p>After adding images, compare candidate and active model scores from the same new run. Older runs may use different validation images.</p>
          </div>
          <div className="training-choice">
            <label htmlFor="trainer">Choose how to train</label>
            <select id="trainer" value={trainer} onChange={e => setTrainer(e.target.value as Trainer)} disabled={!!busy || queued}>
              <option value="baseline">Lightweight baseline</option>
              <option value="vit">Pretrained Vision Transformer</option>
            </select>
            <p>{trainer === "vit" ? "Reuses a pretrained transformer's visual features and trains a new clothing classifier. The transformer stays frozen. The first run downloads model weights; training and predictions run on this computer." : "Trains a new classifier from scratch using simple image features. A quick starting point for comparing experiments."}</p>
            <p>Each run creates a separate version from your labeled images. A higher score is not guaranteed. Your active model stays in use until you activate a passing candidate.</p>
          </div>
          <div className="training-overview"><div><Database size={22} /><span><strong>{overview?.split_counts.train ?? 0}</strong> Training images</span></div><div><CheckCheck size={22} /><span><strong>{overview?.split_counts.validation ?? 0}</strong> Validation images</span></div><div><Layers3 size={22} /><span><strong>{overview?.split_counts.test ?? 0}</strong> Reserved test images</span></div><div><span className={`status-dot ${workerReady ? "online" : ""}`} /><span>{workerReady ? "Worker ready" : "Worker offline"}<small>One run at a time</small></span></div></div>
          <div className="pipeline-steps">{["Prepare", "Train", "Evaluate", "Compare", "Save"].map((step, index) => <span key={step}><span>{index + 1}</span>{step}{index < 4 && <ChevronRight size={16} />}</span>)}</div>
          {!runs.length ? <Empty icon={<FlaskConical size={34} />} title="Your first experiment awaits" text="Import a sample of the fashion dataset, then start a local training run. A lightweight CPU baseline keeps the first experiment small." action="View dataset setup" onAction={() => setSetupOpen(true)} /> : <div className="run-list">{runs.map(run => <article className="run-card" key={run.id}><div className="run-header"><div><h2>Run {run.id.slice(0, 6)}</h2><p>{date(run.created_at)} · {run.trainer === "vit" ? "Vision Transformer" : "Lightweight baseline"}</p></div><span className={`status-tag ${run.status}`}>{run.status}</span></div><div className="run-progress-label"><span>{run.step}</span><span>{run.progress}%</span></div><div className="run-progress"><div style={{ width: `${run.progress}%` }} /></div>{run.error && <p className="run-error">{run.error}</p>}{run.metrics && <div className="run-metrics"><span>Accuracy <strong>{percent(run.metrics.accuracy)}</strong></span><span>Candidate F1 <strong>{percent(run.metrics.macro_f1)}</strong></span>{run.baseline_metrics && <span>Active model F1 · same data <strong>{percent(run.baseline_metrics.macro_f1)}</strong></span>}<span>Validation images <strong>{run.metrics.samples}</strong></span><span>Comparison <strong>{run.metrics.gate_passed ? "Passed" : "Below threshold"}</strong></span><button className="text-button" onClick={() => setView("models")}>View model <ChevronRight size={15} /></button></div>}</article>)}</div>}
          <p className="footnote">First-model threshold: 20% macro F1. Later candidates must improve by at least 1 percentage point on the same validation snapshot. These are learning-demo rules, not a production quality guarantee.</p>
        </>}

        {view === "models" && <>
          {!models.length ? <Empty icon={<Layers3 size={34} />} title="A home for every model version" text="Completed runs appear here with real evaluation results. Activate a passing candidate when you are ready to use it." action="Go to training" onAction={() => setView("training")} /> : <div className="model-list">{models.map(model => <article className={`model-card ${model.active ? "is-active" : ""}`} key={model.id}><div className="run-header"><div><div className="model-name"><h2>{model.id}</h2>{model.active && <span className="status-tag completed">Active</span>}</div><p>{model.metrics.algorithm}</p></div><button className={model.active ? "secondary" : "primary"} disabled={model.active || !model.can_activate || !!busy} onClick={() => void activate(model.id)}>{model.active ? <Check size={16} /> : model.previously_active ? <RotateCcw size={16} /> : <Sparkles size={16} />}{model.active ? "In use" : model.previously_active ? "Restore version" : "Activate model"}</button></div><div className="model-metrics"><div><span>Validation accuracy</span><strong>{percent(model.metrics.accuracy)}</strong></div><div><span>Validation macro F1</span><strong>{percent(model.metrics.macro_f1)}</strong></div><div><span>Training images</span><strong>{model.metrics.training_samples}</strong></div><div><span>Quality gate</span><strong className="gate-value">{model.eligible ? "Passed" : "Not passed"}</strong></div></div><details><summary>Inspect validation confusion matrix</summary><p className="muted">Rows are actual labels; columns are predictions. Larger diagonal values are better.</p><div className="table-scroll"><table><thead><tr><th>Actual / predicted</th>{model.metrics.labels.map(l => <th key={l}>{l}</th>)}</tr></thead><tbody>{model.metrics.confusion_matrix.map((row, i) => <tr key={i}><th>{model.metrics.labels[i]}</th>{row.map((n, j) => <td className={i === j ? "diagonal" : ""} key={j}>{n}</td>)}</tr>)}</tbody></table></div></details><TestReportPanel model={model} busy={!!busy} onRequest={id => void requestTestReport(id)} /></article>)}</div>}
        </>}
        <footer className="page-footer"><span>Built to learn. Designed to improve.</span><span>Fashion dataset by fnauman · CC BY 4.0</span></footer>
      </main>
    </div>
    {setupOpen && <Setup onClose={() => setSetupOpen(false)} />}
  </div>;
}

function Empty({ icon, title, text, action, onAction }: { icon: React.ReactNode; title: string; text: string; action: string; onAction: () => void }) {
  return <div className="empty-state"><div className="empty-icon">{icon}</div><h2>{title}</h2><p>{text}</p><button className="primary" onClick={onAction}>{action}<ChevronRight size={16} /></button></div>;
}

function Setup({ onClose }: { onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => { dialog.current?.showModal(); }, []);
  return <dialog className="setup-dialog" ref={dialog} onCancel={onClose} onClick={e => { if (e.target === dialog.current) onClose(); }} aria-labelledby="setup-title"><button className="dialog-close icon-button" onClick={onClose} aria-label="Close setup"><X size={20} /></button><div className="empty-icon"><ImagePlus size={28} /></div><h2 id="setup-title">Start your learning loop</h2><p>Run these commands from the project folder after installing dependencies in the README.</p><ol><li><strong>Start the workspace</strong><code>npm run dev</code><span>Opens the app, prediction API, and training worker.</span></li><li><strong>Import a small fashion sample</strong><code>npm run data:import -- --per-class 30</code><span>Downloads up to 150 images from Hugging Face. Internet access is needed.</span></li><li><strong>Train, inspect, activate</strong><span>Open Training runs, start a run, then activate an eligible candidate in Model library.</span></li></ol><div className="setup-note">The first classifier is a lightweight CPU baseline. No AWS resources are created. SageMaker reference code is separate.</div></dialog>;
}
