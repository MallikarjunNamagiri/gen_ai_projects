"use client";
import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { 
  streamChatResponse, 
  uploadDocument, 
  fetchAvailableDocuments, 
  selectDocument, 
  fetchSystemStatus, 
  evaluateTurn,
  SystemStatus,
  RagasMetrics
} from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface DocInfo {
  name: string;
  hash: string;
  chunks: number;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  cached?: boolean;
  sources?: any[];
  contexts?: string[];
  evalMetrics?: RagasMetrics;
  isEvaluating?: boolean;
}

export default function DocumentAIChat() {
  const [mounted, setMounted] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // Status & Docs
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [availableDocs, setAvailableDocs] = useState<string[]>([]);
  const [currentDoc, setCurrentDoc] = useState<DocInfo | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);

  const refreshTelemetry = async () => {
    try {
      const [docsData, statusData] = await Promise.all([
        fetchAvailableDocuments(),
        fetchSystemStatus()
      ]);
      setAvailableDocs(docsData.files || []);
      setStatus(statusData);
    } catch {
      // Backend polling
    }
  };

  useEffect(() => {
    setMounted(true);
    refreshTelemetry();
  }, []);

  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [messages]);

  if (!mounted) {
    return (
      <div className="flex h-screen bg-[#080a0e] text-white">
        <aside className="w-80 border-r border-white/10 p-5" />
        <main className="flex-1 max-w-4xl mx-auto p-6" />
      </div>
    );
  }

  const handleSelectExisting = async (filename: string) => {
    if (!filename || isProcessing) return;
    setIsProcessing(true);
    setErrorMessage(null);
    try {
      const data = await selectDocument(filename);
      setCurrentDoc(data);
      await refreshTelemetry();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Loaded **${data.name}** (${data.chunks} chunks). Ready for questions!`,
          cached: false,
          sources: [],
        },
      ]);
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to load document");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsProcessing(true);
    setErrorMessage(null);

    try {
      const data = await uploadDocument(file);
      setCurrentDoc(data);
      await refreshTelemetry();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Successfully uploaded and indexed **${data.name}** (${data.chunks} chunks).`,
          cached: false,
          sources: [],
        },
      ]);
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to upload document");
    } finally {
      setIsProcessing(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const runEvaluationForTurn = async (index: number, question: string, answer: string, contexts: string[]) => {
    setMessages((prev) => {
      const updated = [...prev];
      if (updated[index]) updated[index].isEvaluating = true;
      return updated;
    });

    try {
      const metrics = await evaluateTurn(question, answer, contexts);
      setMessages((prev) => {
        const updated = [...prev];
        if (updated[index]) {
          updated[index].evalMetrics = metrics;
          updated[index].isEvaluating = false;
        }
        return updated;
      });
    } catch {
      setMessages((prev) => {
        const updated = [...prev];
        if (updated[index]) updated[index].isEvaluating = false;
        return updated;
      });
    }
  };

  const submitQuery = async (queryText: string) => {
    if (!queryText.trim() || loading) return;

    setInput("");
    const userMsg: ChatMessage = { role: "user", content: queryText };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    let assistantText = "";
    const currentAssistantIdx = messages.length + 1;
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", cached: false, sources: [], contexts: [] }
    ]);

    try {
      await streamChatResponse(
        queryText,
        currentDoc ? currentDoc.hash : "",
        messages,
        (token) => {
          assistantText += token;
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1].content = assistantText;
            return next;
          });
        },
        async (meta) => {
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1].cached = meta.cached;
            next[next.length - 1].sources = meta.sources;
            next[next.length - 1].contexts = meta.contexts || [];
            return next;
          });
          setLoading(false);
          refreshTelemetry();

          // Auto-trigger Ragas evaluation if sources/contexts exist and not from cache
          if (!meta.cached && meta.contexts?.length > 0) {
            runEvaluationForTurn(currentAssistantIdx, queryText, assistantText, meta.contexts);
          }
        }
      );
    } catch (err: any) {
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1].content = `⚠️ ${err.message || "Request failed."}`;
        return next;
      });
      setLoading(false);
    }
  };

  const extractFollowUps = (text: string) => {
    const followUpMarker = "**Suggested Follow-ups:**";
    if (!text.includes(followUpMarker)) return { mainText: text, followUps: [] };

    const [mainText, followUpBlock] = text.split(followUpMarker);
    const followUps = followUpBlock
      .split("\n")
      .map((line) => line.replace(/^[-*•\d.]\s*/, "").trim())
      .filter((line) => line.length > 5);

    return { mainText, followUps };
  };

  return (
    <div className="flex h-screen bg-[#080a0e] text-white" suppressHydrationWarning>
      {/* Sidebar */}
      <aside className="w-80 border-r border-white/10 p-5 flex flex-col justify-between overflow-y-auto">
        <div>
          {/* Header */}
          <div className="flex items-center gap-3 pb-4 border-b border-white/10">
            <span className="text-amber-400 text-2xl font-bold">✦</span>
            <span className="font-semibold tracking-wide text-lg">Document AI</span>
          </div>

          {/* ENGINE & STATUS CARD */}
          <div className="mt-5">
            <div className="flex items-center gap-1.5 text-xs uppercase font-bold text-gray-400 tracking-wider mb-2.5">
              <span className="text-amber-500">⚡</span> ENGINE & STATUS
            </div>
            <div className="bg-[#10141b] border border-white/10 rounded-2xl p-4 space-y-3 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">LLM Provider</span>
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-gray-200">{status?.llm_provider || "Groq"}</span>
                  <span className="bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded-full text-[10px] flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span> Connected
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-gray-400">Model</span>
                <span className="font-mono text-[11px] text-gray-200 truncate max-w-[140px]" title={status?.model}>
                  {status?.model || "llama-3.3-70b"}
                </span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-gray-400">Doc Vector DB</span>
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-gray-200">{status?.vector_db || "FAISS"}</span>
                  <span className={`px-2 py-0.5 rounded-full text-[10px] flex items-center gap-1 ${
                    currentDoc 
                      ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30" 
                      : "bg-white/5 text-gray-400 border border-white/10"
                  }`}>
                    <span className={`w-1.5 h-1.5 rounded-full ${currentDoc ? "bg-emerald-400" : "bg-gray-400"}`}></span>
                    {currentDoc ? "Ready" : "Waiting File"}
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-gray-400">Evaluation Engine</span>
                <span className="font-semibold text-amber-400 font-mono text-[11px]">RAGAS Active</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-gray-400">Retrieval</span>
                <span className="font-semibold text-gray-200">{status?.retrieval || "Hybrid + Router"}</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-gray-400">Guardrail</span>
                <span className="font-mono text-gray-200">{status?.guardrail || "Cutoff (-2.5)"}</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-gray-400">Semantic Cache</span>
                <span className="font-mono text-gray-200">{status?.semantic_cache_count || "0 cached"}</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-gray-400">Match Threshold</span>
                <span className="font-mono text-gray-200">{status?.match_threshold || "88%"}</span>
              </div>
            </div>
          </div>

          {/* Active Document Card */}
          <div className="mt-5">
            <div className="text-xs uppercase font-bold text-gray-400 tracking-wider mb-2">Active Document</div>
            {currentDoc ? (
              <div className="bg-[#11151b] border border-amber-400/20 rounded-xl p-3.5 space-y-1.5">
                <div className="flex items-start gap-2">
                  <span className="text-amber-400 text-sm">📄</span>
                  <div className="overflow-hidden">
                    <p className="text-sm font-medium text-gray-100 truncate" title={currentDoc.name}>
                      {currentDoc.name}
                    </p>
                    <p className="text-[11px] text-gray-400 font-mono">{currentDoc.chunks} Chunks Indexed</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-[#11151b] border border-dashed border-white/15 rounded-xl p-3.5 text-center">
                <p className="text-xs text-gray-400">No document active</p>
              </div>
            )}
          </div>

          {/* Available Backend Documents */}
          {availableDocs.length > 0 && (
            <div className="mt-4">
              <label className="text-[11px] uppercase font-bold text-gray-400 tracking-wider block mb-1.5">
                Available in Backend Data
              </label>
              <select
                disabled={isProcessing}
                onChange={(e) => handleSelectExisting(e.target.value)}
                defaultValue=""
                className="w-full bg-[#11151b] border border-white/15 text-xs text-gray-200 rounded-lg p-2.5 outline-none focus:border-amber-400/50"
              >
                <option value="" disabled>Select existing PDF...</option>
                {availableDocs.map((doc, idx) => (
                  <option key={idx} value={doc}>{doc}</option>
                ))}
              </select>
            </div>
          )}

          {/* Upload Button */}
          <div className="mt-4">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept=".pdf"
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isProcessing}
              className={`w-full py-2.5 px-3 rounded-xl text-xs font-semibold flex items-center justify-center gap-2 transition ${
                isProcessing
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/30 cursor-wait"
                  : "bg-[#1a1f28] hover:bg-[#232936] text-amber-400 border border-amber-400/30 hover:border-amber-400/60"
              }`}
            >
              {isProcessing ? (
                <>
                  <span className="animate-spin text-sm">⟳</span> Processing...
                </>
              ) : (
                <>
                  <span>⇪</span> Upload New PDF
                </>
              )}
            </button>
            {errorMessage && (
              <p className="text-red-400 text-[11px] mt-2 leading-tight">{errorMessage}</p>
            )}
          </div>
        </div>

        <div className="text-xs text-gray-500 border-t border-white/10 pt-4 flex justify-between items-center mt-6">
          <span>Enterprise RAG v2.0</span>
          <span className="text-[10px] bg-white/5 px-2 py-0.5 rounded text-gray-400">FastAPI</span>
        </div>
      </aside>

      {/* Main Chat Interface */}
      <main className="flex-1 flex flex-col max-w-4xl mx-auto p-6 w-full">
        <div ref={chatScrollRef} className="flex-1 overflow-y-auto space-y-4 pr-3" suppressHydrationWarning>
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-center opacity-60">
              <span className="text-amber-400 text-4xl mb-3">✦</span>
              <h3 className="text-base font-medium text-gray-200">Document AI Platform</h3>
              <p className="text-xs text-gray-400 max-w-sm mt-1">
                Select an existing document from the dropdown or upload a new PDF to start querying.
              </p>
            </div>
          )}

          {messages.map((m, idx) => {
            const { mainText, followUps } = m.role === "assistant" 
              ? extractFollowUps(m.content) 
              : { mainText: m.content, followUps: [] };

            return (
              <div key={idx} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`p-4 rounded-xl max-w-[85%] ${
                    m.role === "user"
                      ? "bg-[#1a1f28] text-gray-100 border border-white/5"
                      : "bg-[#11151b] border border-white/10"
                  }`}
                >
                  {m.role === "assistant" && (
                    <div className="text-xs text-amber-400 font-bold mb-2 flex items-center justify-between">
                      <span>✦ DOCUMENT AI</span>
                      {m.cached && (
                        <span className="bg-green-500/20 border border-green-500/30 text-green-400 px-2 py-0.5 rounded text-[10px]">
                          ⚡ Cache Hit
                        </span>
                      )}
                    </div>
                  )}

                  {/* Markdown Response Body */}
                  <div className="text-sm leading-relaxed text-gray-200 space-y-2">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        table: ({ children }) => (
                          <div className="my-3 overflow-x-auto rounded-lg border border-white/10">
                            <table className="min-w-full divide-y divide-white/10 text-xs text-left">
                              {children}
                            </table>
                          </div>
                        ),
                        thead: ({ children }) => <thead className="bg-[#1a1f28] text-amber-300 font-semibold">{children}</thead>,
                        tbody: ({ children }) => <tbody className="divide-y divide-white/5 bg-[#141820]">{children}</tbody>,
                        tr: ({ children }) => <tr className="hover:bg-white/[0.02] transition">{children}</tr>,
                        th: ({ children }) => <th className="px-3.5 py-2.5 font-semibold text-gray-200 border-r border-white/10 last:border-r-0">{children}</th>,
                        td: ({ children }) => <td className="px-3.5 py-2 text-gray-300 border-r border-white/5 last:border-r-0 align-top">{children}</td>,
                        ul: ({ children }) => <ul className="list-disc pl-5 space-y-1 my-2">{children}</ul>,
                        ol: ({ children }) => <ol className="list-decimal pl-5 space-y-1 my-2">{children}</ol>,
                        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                        strong: ({ children }) => <strong className="font-semibold text-white">{children}</strong>,
                      }}
                    >
                      {mainText}
                    </ReactMarkdown>
                  </div>

                  {/* RAGAS EVALUATION MATRIX BLOCK */}
                  {m.role === "assistant" && !m.cached && (m.evalMetrics || m.isEvaluating) && (
                    <div className="mt-4 pt-3 border-t border-white/10">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-[11px] uppercase tracking-wider font-bold text-amber-400/90 flex items-center gap-1.5">
                          <span>📊</span> RAGAS Evaluation Matrix
                        </span>
                        {m.isEvaluating && (
                          <span className="text-[10px] text-gray-400 font-mono flex items-center gap-1">
                            <span className="animate-spin text-amber-400">⟳</span> Scoring RAG Triad...
                          </span>
                        )}
                      </div>

                      {m.evalMetrics && (
                        <div className="grid grid-cols-3 gap-2 mt-1">
                          <div className="bg-[#151a23] border border-white/5 rounded-lg p-2 text-center">
                            <div className="text-[10px] text-gray-400">Faithfulness</div>
                            <div className="text-sm font-mono font-bold text-emerald-400 mt-0.5">
                              {(m.evalMetrics.faithfulness * 100).toFixed(0)}%
                            </div>
                            <div className="text-[9px] text-gray-500">Groundedness</div>
                          </div>

                          <div className="bg-[#151a23] border border-white/5 rounded-lg p-2 text-center">
                            <div className="text-[10px] text-gray-400">Context Relevancy</div>
                            <div className="text-sm font-mono font-bold text-sky-400 mt-0.5">
                              {(m.evalMetrics.context_relevancy * 100).toFixed(0)}%
                            </div>
                            <div className="text-[9px] text-gray-500">Retriever Signal</div>
                          </div>

                          <div className="bg-[#151a23] border border-white/5 rounded-lg p-2 text-center">
                            <div className="text-[10px] text-gray-400">Answer Correctness</div>
                            <div className="text-sm font-mono font-bold text-amber-400 mt-0.5">
                              {(m.evalMetrics.answer_correctness * 100).toFixed(0)}%
                            </div>
                            <div className="text-[9px] text-gray-500">Semantic Alignment</div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Clickable Follow-up Question Chips */}
                  {followUps.length > 0 && (
                    <div className="mt-4 pt-3 border-t border-white/10">
                      <div className="text-[11px] font-semibold text-amber-400/90 mb-2 flex items-center gap-1.5">
                        <span>💬</span> Suggested Follow-ups
                      </div>
                      <div className="flex flex-col gap-1.5">
                        {followUps.map((question, qIdx) => (
                          <button
                            key={qIdx}
                            onClick={() => submitQuery(question)}
                            disabled={loading}
                            className="text-left text-xs bg-[#191f2b] hover:bg-[#222a3a] border border-white/10 hover:border-amber-400/40 text-gray-300 hover:text-amber-300 px-3 py-2 rounded-lg transition"
                          >
                            → {question}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Clickable Citations */}
                  {m.sources && m.sources.length > 0 && (
                    <div className="mt-3.5 pt-2.5 border-t border-white/10 text-xs text-gray-400 flex flex-wrap items-center gap-1.5">
                      <span className="font-semibold text-gray-300 mr-1">Sources:</span>
                      {m.sources.map((s: any, sIdx: number) => {
                        const fileUrl = `${API_BASE}/files/${encodeURIComponent(s.source)}#page=${s.page}`;
                        return (
                          <a
                            key={sIdx}
                            href={fileUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="bg-white/5 hover:bg-amber-400/10 border border-white/10 hover:border-amber-400/40 text-gray-300 hover:text-amber-300 px-2 py-0.5 rounded text-[11px] transition inline-flex items-center gap-1"
                            title={`Open ${s.source} at page ${s.page}`}
                          >
                            <span>📄 {s.source}</span>
                            <span className="text-amber-400 font-mono">(p.{s.page})</span>
                            <span className="text-[9px] opacity-60">↗</span>
                          </a>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Input Field */}
        <div className="mt-4 flex gap-2">
          <input
            className="flex-1 bg-[#1a1f28] border border-white/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-amber-400/50 placeholder-gray-500"
            placeholder={currentDoc ? `Ask a question about ${currentDoc.name}...` : "Upload or select a document first..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submitQuery(input)}
          />
          <button
            onClick={() => submitQuery(input)}
            disabled={loading || !input.trim()}
            className="bg-amber-500 hover:bg-amber-600 disabled:opacity-50 disabled:cursor-not-allowed text-black font-semibold px-6 py-3 rounded-xl text-sm transition"
          >
            {loading ? "..." : "Send"}
          </button>
        </div>
      </main>
    </div>
  );
}