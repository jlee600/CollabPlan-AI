import React, { useRef, useState } from "react";

/**
 * CollabPlan UI – full single-file app
 * - Upload audio -> /transcribe_upload
 * - Record audio (mic) -> /transcribe_upload
 * - Server-path transcribe -> /transcribe
 * - Analyze pasted text -> /analyze
 * - Analyze uploaded text file -> /analyze_file
 * - View/Analyze run -> /run/{id}, /analyze/{id}
 * - List/Export runs -> /runs, /export/{id}.csv|.md
 */

// ---------- Types ----------
type Tab = "Transcribe" | "Analyze text" | "Run view" | "Runs";

interface SegmentOut {
  idx: number;
  start: number;
  end: number;
  text: string;
  speaker?: string | null;
}

interface TranscribeResponse {
  run_id: string;
  duration_sec: number;
  segments: SegmentOut[];
}

interface Task {
  title: string;
  owner?: string | null;
  due_date?: string | null; // ISO
  priority?: string | null; // "High" | "Medium" | "Low" | null
  dependencies?: string[];
  confidence?: number | null;
}

interface AnalyzeTextResponse {
  run_id: string;
  meeting_date: string; // ISO
  summary: string;
  tasks: Task[];
  open_questions: string[];
}

interface RunBundle {
  run_id: string;
  meeting_date: string;
  duration_sec?: number | null;
  segments: SegmentOut[];
  summary: string;
  tasks: Task[];
  open_questions: string[];
}

interface RunListItem {
  run_id: string;
  meeting_date: string;
  duration_sec?: number | null;
  summary_preview?: string;
  task_count?: number;
}

// ---------- Styles ----------
const section = "max-w-[1200px] mx-auto px-4 py-6";
const card =
  "rounded-2xl shadow p-4 bg-white border border-gray-100";
const h2 = "text-xl font-semibold mb-3";
const label = "text-sm font-medium text-gray-700";
const input =
  "w-full rounded-lg border px-3 py-2 focus:outline-none focus:ring focus:border-gray-400";
const btn =
  "inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium border border-gray-300 hover:bg-gray-50 disabled:opacity-50";
const btnPrimary =
  "inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium bg-black text-white hover:opacity-90 disabled:opacity-50";
const pill =
  "px-2.5 py-0.5 rounded-full text-xs bg-gray-100 text-gray-700";

// ---------- Helpers ----------
function fmtDateISO(d: string | Date): string {
  const dt = typeof d === "string" ? new Date(d) : d;
  const y = dt.getFullYear();
  const m = String(dt.getMonth() + 1).padStart(2, "0");
  const da = String(dt.getDate()).padStart(2, "0");
  return `${y}-${m}-${da}`;
}

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
      />
    </svg>
  );
}

async function callJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, {
    ...init,
    headers: { ...(init?.headers || {}) },
  });
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(`${r.status} ${r.statusText} - ${txt}`);
  }
  return r.json() as Promise<T>;
}

function asNumberOrDash(v: number | null | undefined, digits = 1): string {
  if (typeof v === "number" && Number.isFinite(v)) return v.toFixed(digits);
  return "-";
}

function pickRecordingMime(): string {
  // Prefer Opus in WebM on Chromium, fallback gracefully
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/mpeg",
  ];
  const isSup = (t: string) => (window.MediaRecorder?.isTypeSupported?.(t) ?? false);
  for (const mt of candidates) if (isSup(mt)) return mt;
  return "audio/webm";
}

// ---------- App ----------
export default function App() {
  const [apiBase, setApiBase] = useState("http://localhost:8000");

  // Tabs
  const tabs: Tab[] = ["Transcribe", "Analyze text", "Run view", "Runs"];
  const [tab, setTab] = useState<Tab>("Transcribe");

  // Shared
  const [meetingDate, setMeetingDate] = useState(fmtDateISO(new Date()));
  const [lastRunId, setLastRunId] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  // Transcribe (server path / upload / record)
  const [path, setPath] = useState("data/samples/sample2.mp3");
  const [source, setSource] = useState<"upload" | "recording">("upload");
  const [diarize, setDiarize] = useState(true);
  const [transcribeResp, setTranscribeResp] = useState<TranscribeResponse | null>(null);

  // Upload
  const [uploadFile, setUploadFile] = useState<File | null>(null);

  // Recording
  const [isRecording, setIsRecording] = useState(false);
  const [recError, setRecError] = useState<string>("");
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recStreamRef = useRef<MediaStream | null>(null);

  // Analyze text/file
  const [rawText, setRawText] = useState(
    "Tom: Ship /v1/auth by Friday.\nSameer: Draft the API guide.\nWill: Prepare a demo next week."
  );
  const [fileToAnalyze, setFileToAnalyze] = useState<File | null>(null);
  const [analyzeTextResp, setAnalyzeTextResp] = useState<AnalyzeTextResponse | null>(null);

  // Run view
  const [runIdInput, setRunIdInput] = useState("");
  const [runBundle, setRunBundle] = useState<RunBundle | null>(null);

  // Runs list
  const [runs, setRuns] = useState<RunListItem[]>([]);

  function exportHref(kind: "csv" | "md") {
    if (!lastRunId) return "#";
    return `${apiBase}/export/${lastRunId}.${kind}`;
  }

  // ---------- Actions ----------
  async function doTranscribe(): Promise<void> {
    setBusy(true);
    setMsg("");
    setTranscribeResp(null);
    try {
      const body = {
        meeting_date: meetingDate,
        source, // will be "upload" or "recording" or whatever you set – ok to pass through
        path,
        diarize,
      };
      const data = await callJSON<TranscribeResponse>(`${apiBase}/transcribe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setTranscribeResp(data);
      setLastRunId(data.run_id);
      setRunIdInput(data.run_id);
      setTab("Run view");
      setMsg("Transcribed");
    } catch (e: any) {
      setMsg(e.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  async function doUploadAndTranscribe(): Promise<void> {
    if (source !== "upload") {
      setMsg('Set Source to "upload" to enable file uploads.');
      return;
    }
    if (!uploadFile) return;
    setBusy(true);
    setMsg("");
    setTranscribeResp(null);
    try {
      const fd = new FormData();
      fd.append("meeting_date", meetingDate);
      fd.append("diarize", String(diarize));
      fd.append("file", uploadFile, uploadFile.name || "audio");
      const r = await fetch(`${apiBase}/transcribe_upload`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      const data = (await r.json()) as TranscribeResponse;
      setTranscribeResp(data);
      setLastRunId(data.run_id);
      setRunIdInput(data.run_id);
      setTab("Run view");
      setMsg("Uploaded & transcribed");
    } catch (e: any) {
      setMsg(e.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  async function startRecording(): Promise<void> {
    if (source !== "recording") {
      setRecError('Set Source to "recording" first.');
      return;
    }
    setRecError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = pickRecordingMime();
      const mr = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mr;
      recStreamRef.current = stream;

      mr.start();
      setIsRecording(true);
      setMsg("Recording...");
    } catch (e: any) {
      setRecError(e.message || String(e));
      setIsRecording(false);
    }
  }

  async function stopRecordingAndTranscribe(): Promise<void> {
    if (source !== "recording") {
      setRecError('Set Source to "recording" first.');
      return;
    }
    const mr = mediaRecorderRef.current;
    if (!mr) return;

    setBusy(true);
    setMsg("");
    setTranscribeResp(null);

    try {
      const blob: Blob = await new Promise((resolve, reject) => {
        const chunks: BlobPart[] = [];
        mr.ondataavailable = (ev: BlobEvent) => {
          if (ev.data && ev.data.size > 0) chunks.push(ev.data);
        };
        mr.onstop = () => {
          try {
            resolve(new Blob(chunks, { type: mr.mimeType || "audio/webm" }));
          } catch (err) {
            reject(err);
          }
        };
        mr.onerror = (e) => reject(e);
        mr.stop();
      });

      // Stop tracks
      recStreamRef.current?.getTracks().forEach((t) => t.stop());
      mediaRecorderRef.current = null;
      recStreamRef.current = null;
      setIsRecording(false);

      // Send blob directly to /transcribe_upload
      const filename =
        (blob.type.includes("mp4") && "recording.mp4") ||
        (blob.type.includes("mpeg") && "recording.mp3") ||
        "recording.webm";

      const fd = new FormData();
      fd.append("meeting_date", meetingDate);
      fd.append("diarize", String(diarize));
      fd.append("file", blob, filename);

      const r = await fetch(`${apiBase}/transcribe_upload`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      const data = (await r.json()) as TranscribeResponse;

      setTranscribeResp(data);
      setLastRunId(data.run_id);
      setRunIdInput(data.run_id);
      setTab("Run view");
      setMsg("Recorded & transcribed");
    } catch (e: any) {
      setRecError(e.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  async function doAnalyzeRun(runId: string): Promise<void> {
    if (!runId) return;
    setBusy(true);
    setMsg("");
    try {
      await callJSON<AnalyzeTextResponse>(`${apiBase}/analyze/${runId}`, { method: "POST" });
      setLastRunId(runId);
      setMsg("Analyzed");
      await loadRun(runId);
    } catch (e: any) {
      setMsg(e.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  async function doAnalyzeText(): Promise<void> {
    setBusy(true);
    setMsg("");
    setAnalyzeTextResp(null);
    try {
      const body = { meeting_date: meetingDate, transcript: rawText };
      const data = await callJSON<AnalyzeTextResponse>(`${apiBase}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setAnalyzeTextResp(data);
      setLastRunId(data.run_id);
      setRunIdInput(data.run_id);
      setTab("Run view");
      setMsg("Analyzed text");
    } catch (e: any) {
      setMsg(e.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  async function doAnalyzeFile(): Promise<void> {
    if (!fileToAnalyze) return;
    setBusy(true);
    setMsg("");
    setAnalyzeTextResp(null);
    try {
      const fd = new FormData();
      fd.append("meeting_date", meetingDate);
      fd.append("file", fileToAnalyze, fileToAnalyze.name || "notes.txt");
      const r = await fetch(`${apiBase}/analyze_file`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      const data = (await r.json()) as AnalyzeTextResponse;
      setAnalyzeTextResp(data);
      setLastRunId(data.run_id);
      setRunIdInput(data.run_id);
      setTab("Run view");
      setMsg("Analyzed file");
    } catch (e: any) {
      setMsg(e.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  async function loadRun(runId: string): Promise<void> {
    if (!runId) return;
    setBusy(true);
    setMsg("");
    setRunBundle(null);
    try {
      const data = await callJSON<RunBundle>(`${apiBase}/run/${runId}`);
      setRunBundle(data);
      setMsg("Loaded run");
    } catch (e: any) {
      setMsg(e.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  async function loadRuns(): Promise<void> {
    setBusy(true);
    setMsg("");
    try {
      const data = await callJSON<RunListItem[]>(`${apiBase}/runs`);
      setRuns(data);
      setMsg("Loaded runs");
    } catch (e: any) {
      setMsg(e.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  const tasks = runBundle?.tasks ?? [];
  const segments = runBundle?.segments ?? [];

  // ---------- UI ----------
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b bg-white">
        <div className={`${section} flex items-center justify-between gap-4`}>
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-xl bg-black text-white grid place-items-center font-bold">
              CP
            </div>
            <div>
              <div className="font-semibold">CollabPlan AI</div>
              <div className="text-xs text-gray-500">Powered by Evan Lee</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <input
              className={`${input} w-64`}
              value={apiBase}
              onChange={(e) => setApiBase(e.target.value)}
            />
            <a
              href="/docs"
              className={btn}
              onClick={(e) => {
                e.preventDefault();
                window.open(`${apiBase}/docs`, "_blank");
              }}
            >
              Open Swagger
            </a>
          </div>
        </div>
      </header>

      <main className={section}>
        {/* Tabs */}
        <div className="mb-6 flex flex-wrap gap-2">
          {tabs.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`${tab === t ? "bg-black text-white" : "bg-white text-gray-800"} ${btn}`}
            >
              {t}
            </button>
          ))}
          <div className="ml-auto flex items-center gap-2">
            <input
              type="date"
              className={input}
              style={{ width: 150 }}
              value={meetingDate}
              onChange={(e) => setMeetingDate(e.target.value)}
            />
            {busy ? (
              <span className={pill}>
                <Spinner /> Working
              </span>
            ) : null}
            {msg ? <span className={pill}>{msg}</span> : null}
          </div>
        </div>

        {/* Transcribe */}
        {tab === "Transcribe" && (
          <section className={card}>
            <h2 className={h2}>Transcribe audio</h2>

            {/* Common inputs */}
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <div className={label}>Audio path on server</div>
                <input className={input} value={path} onChange={(e) => setPath(e.target.value)} />
              </div>

              <div>
                <div className={label}>Source</div>
                <select
                  className={input}
                  value={source}
                  onChange={(e) => setSource(e.target.value as "upload" | "recording")}
                >
                  <option value="upload">upload</option>
                  <option value="recording">recording</option>
                </select>
              </div>

              <div className="flex items-center gap-2">
                <input
                  id="diarize"
                  type="checkbox"
                  checked={diarize}
                  onChange={(e) => setDiarize(e.target.checked)}
                />
                <label htmlFor="diarize" className={label}>
                  Diarize
                </label>
              </div>

              {/* Upload input only in upload mode */}
              {source === "upload" && (
                <div>
                  <div className={label}>Upload audio</div>
                  <input
                    type="file"
                    accept="audio/*,video/webm,video/mp4"
                    onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                  />
                </div>
              )}
            </div>

            {/* Recording controls only in recording mode */}
            {source === "recording" && (
              <div className="mt-4">
                <div className={label}>Record with microphone</div>
                <div className="flex gap-2">
                  {!isRecording ? (
                    <button className={btn} onClick={startRecording} disabled={busy}>
                      Start recording
                    </button>
                  ) : (
                    <button className={btnPrimary} onClick={stopRecordingAndTranscribe} disabled={busy}>
                      Stop &amp; Transcribe
                    </button>
                  )}
                  {recError ? <span className={pill}>{recError}</span> : null}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Works on https or http://localhost. Chrome/Edge record WebM/Opus; Safari records MP4/M4A.
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="mt-4 flex flex-wrap gap-2">
              {/* Server-path transcribe is always available */}
              <button className={btnPrimary} onClick={doTranscribe} disabled={busy}>
                Transcribe (server path)
              </button>

              {/* Upload & Transcribe button only if source=upload */}
              {source === "upload" && (
                <button
                  className={btn}
                  onClick={doUploadAndTranscribe}
                  disabled={busy || !uploadFile}
                >
                  Upload &amp; Transcribe
                </button>
              )}

              {lastRunId ? (
                <button className={btn} onClick={() => setTab("Run view")}>
                  Go to run
                </button>
              ) : null}
            </div>

            {/* Teaser */}
            {transcribeResp && (
              <div className="mt-6">
                <div className="text-sm text-gray-600">Run id</div>
                <div className="font-mono text-sm">{transcribeResp.run_id}</div>
                <div className="mt-2 text-sm text-gray-600">First 3 segments</div>
                <ul className="mt-1 space-y-1 text-sm">
                  {transcribeResp.segments.slice(0, 3).map((s) => (
                    <li key={s.idx}>
                      <span className="font-mono text-gray-500">[{s.idx}]</span>{" "}
                      {s.speaker ? <b>{s.speaker}:</b> : null} {s.text}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </section>
        )}

        {/* Analyze text / file */}
        {tab === "Analyze text" && (
          <section className={card}>
            <h2 className={h2}>Analyze pasted text</h2>
            <textarea
              className={`${input} h-48 font-mono`}
              value={rawText}
              onChange={(e) => setRawText(e.target.value)}
            />
            <div className="mt-4 flex gap-2">
              <button className={btnPrimary} onClick={doAnalyzeText} disabled={busy}>
                Analyze text
              </button>
              {analyzeTextResp ? (
                <button className={btn} onClick={() => setTab("Run view")}>
                  Go to run
                </button>
              ) : null}
            </div>

            <div className="mt-6 grid md:grid-cols-2 gap-4">
              <div>
                <div className={label}>Or analyze a .txt/.md file</div>
                <input
                  type="file"
                  accept=".txt,.md,text/plain,text/markdown"
                  onChange={(e) => setFileToAnalyze(e.target.files?.[0] || null)}
                />
                <div className="mt-2">
                  <button className={btn} onClick={doAnalyzeFile} disabled={busy || !fileToAnalyze}>
                    Analyze file
                  </button>
                </div>
              </div>

              {analyzeTextResp && (
                <div>
                  <div className="text-sm text-gray-600">Run id</div>
                  <div className="font-mono text-sm">{analyzeTextResp.run_id}</div>
                  <div className="mt-2 text-sm">Summary</div>
                  <p className="text-sm text-gray-800">{analyzeTextResp.summary}</p>
                </div>
              )}
            </div>
          </section>
        )}

        {/* Run view */}
        {tab === "Run view" && (
          <section className={card}>
            <h2 className={h2}>Run view</h2>
            <div className="flex flex-wrap gap-2 items-end mb-4">
              <div>
                <div className={label}>Run id</div>
                <input
                  className={`${input} w-80`}
                  value={runIdInput}
                  onChange={(e) => setRunIdInput(e.target.value)}
                />
              </div>
              <button className={btn} onClick={() => loadRun(runIdInput)} disabled={busy || !runIdInput}>
                Load
              </button>
              <button
                className={btnPrimary}
                onClick={() => doAnalyzeRun(runIdInput)}
                disabled={busy || !runIdInput}
              >
                Analyze
              </button>
              {lastRunId && (
                <>
                  <a className={btn} href={exportHref("csv")} target="_blank" rel="noreferrer">
                    Export CSV
                  </a>
                  <a className={btn} href={exportHref("md")} target="_blank" rel="noreferrer">
                    Export MD
                  </a>
                </>
              )}
            </div>

            {runBundle ? (
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <div className="text-sm text-gray-500">Meta</div>
                  <div className="mt-2 grid grid-cols-2 text-sm">
                    <div>Run id</div>
                    <div className="font-mono">{runBundle.run_id}</div>
                    <div>Date</div>
                    <div>{runBundle.meeting_date}</div>
                    <div>Duration</div>
                    <div>
                      {typeof runBundle.duration_sec === "number"
                        ? `${asNumberOrDash(runBundle.duration_sec, 1)} s`
                        : "-"}
                    </div>
                  </div>

                  <div className="mt-4 text-sm text-gray-500">Summary</div>
                  <p className="text-sm">{runBundle.summary || "(none)"}</p>

                  <div className="mt-4 text-sm text-gray-500">Open questions</div>
                  <ul className="list-disc pl-5 text-sm">
                    {(runBundle.open_questions || []).map((q, i) => (
                      <li key={i}>{q}</li>
                    ))}
                  </ul>
                </div>

                <div>
                  <div className="text-sm text-gray-500 mb-1">Tasks ({tasks.length})</div>
                  <div className="overflow-auto border rounded-xl">
                    <table className="min-w-full text-sm">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="text-left p-2">Title</th>
                          <th className="text-left p-2">Owner</th>
                          <th className="text-left p-2">Due</th>
                          <th className="text-left p-2">Priority</th>
                          <th className="text-left p-2">Deps</th>
                          <th className="text-left p-2">Conf</th>
                        </tr>
                      </thead>
                      <tbody>
                        {tasks.map((t, i) => (
                          <tr key={i} className={i % 2 ? "bg-white" : "bg-gray-50"}>
                            <td className="p-2">{t.title}</td>
                            <td className="p-2">{t.owner || "-"}</td>
                            <td className="p-2">{t.due_date || "-"}</td>
                            <td className="p-2">{t.priority || "-"}</td>
                            <td
                              className="p-2 max-w-[280px] truncate"
                              title={(t.dependencies || []).join(", ")}
                            >
                              {(t.dependencies || []).join("; ")}
                            </td>
                            <td className="p-2">
                              {t.confidence == null ? "-" : Number(t.confidence).toFixed(2)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="md:col-span-2">
                  <div className="text-sm text-gray-500 mb-1">
                    Transcript segments ({segments.length})
                  </div>
                  <div className="max-h-72 overflow-auto border rounded-xl bg-white">
                    <table className="min-w-full text-sm">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="text-left p-2 w-16">#</th>
                          <th className="text-left p-2 w-28">Speaker</th>
                          <th className="text-left p-2">Text</th>
                        </tr>
                      </thead>
                      <tbody>
                        {segments.map((s) => (
                          <tr key={s.idx} className={s.idx % 2 ? "bg-white" : "bg-gray-50"}>
                            <td className="p-2 font-mono text-gray-600">{s.idx}</td>
                            <td className="p-2">{s.speaker || "-"}</td>
                            <td className="p-2">{s.text}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-sm text-gray-600">Load or analyze a run to see details.</p>
            )}
          </section>
        )}

        {/* Runs */}
        {tab === "Runs" && (
          <section className={card}>
            <h2 className={h2}>Runs</h2>
            <div className="flex gap-2 mb-3">
              <button className={btn} onClick={loadRuns} disabled={busy}>
                Refresh
              </button>
              {lastRunId ? (
                <button
                  className={btn}
                  onClick={() => {
                    setRunIdInput(lastRunId);
                    setTab("Run view");
                  }}
                >
                  Open last run
                </button>
              ) : null}
            </div>
            <div className="overflow-auto border rounded-xl">
              <table className="min-w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="text-left p-2">Run id</th>
                    <th className="text-left p-2">Date</th>
                    <th className="text-left p-2">Duration</th>
                    <th className="text-left p-2">Tasks</th>
                    <th className="text-left p-2">Summary</th>
                    <th className="text-left p-2">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r) => (
                    <tr key={r.run_id} className="odd:bg-gray-50">
                      <td className="p-2 font-mono">{r.run_id}</td>
                      <td className="p-2">{r.meeting_date}</td>
                      <td className="p-2">
                        {typeof r.duration_sec === "number"
                          ? `${asNumberOrDash(r.duration_sec, 1)} s`
                          : "-"}
                      </td>
                      <td className="p-2">{r.task_count ?? "-"}</td>
                      <td className="p-2 max-w-[480px] truncate" title={r.summary_preview || ""}>
                        {r.summary_preview || ""}
                      </td>
                      <td className="p-2">
                        <button
                          className={btn}
                          onClick={() => {
                            setRunIdInput(r.run_id);
                            setTab("Run view");
                            void loadRun(r.run_id);
                          }}
                        >
                          Open
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </main>

      <footer className="py-8 text-center text-xs text-gray-400">for all georgia tech yellow jackets</footer>
    </div>
  );
}
