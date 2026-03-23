"use client";

import { useState } from "react";
import { PROMPT_TEMPLATES } from "@/lib/prompts";

export default function ImageGenerator() {
  const [prompt, setPrompt] = useState("");
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [usedPrompt, setUsedPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [approving, setApproving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const generate = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);
    setStatus(null);

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();

      if (!res.ok) {
        setError(data.error || "Generation failed");
        return;
      }

      setGeneratedImage(data.image);
      setUsedPrompt(data.prompt);
    } catch (err: any) {
      setError(err.message || "Network error");
    } finally {
      setLoading(false);
    }
  };

  const approve = async () => {
    if (!generatedImage) return;
    setApproving(true);
    setError(null);

    try {
      const res = await fetch("/api/approve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: generatedImage, prompt: usedPrompt }),
      });
      const data = await res.json();

      if (!res.ok) {
        setError(data.error || "Approval failed");
        return;
      }

      setStatus(`Saved as ${data.metadata.filename}`);
      setGeneratedImage(null);
    } catch (err: any) {
      setError(err.message || "Network error");
    } finally {
      setApproving(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Prompt Input */}
      <div className="space-y-3">
        <label className="block text-sm font-medium text-zinc-400">
          Kitchen Description
        </label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe the kitchen you want to generate..."
          rows={3}
          className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-3 text-white placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
        />

        <div className="flex gap-3">
          <button
            onClick={generate}
            disabled={loading || !prompt.trim()}
            className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 rounded-lg font-medium transition-colors"
          >
            {loading ? "Generating..." : "Generate"}
          </button>
          {generatedImage && (
            <>
              <button
                onClick={approve}
                disabled={approving}
                className="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-700 rounded-lg font-medium transition-colors"
              >
                {approving ? "Saving..." : "Approve & Save"}
              </button>
              <button
                onClick={() => setGeneratedImage(null)}
                className="px-5 py-2.5 bg-zinc-700 hover:bg-zinc-600 rounded-lg font-medium transition-colors"
              >
                Discard
              </button>
            </>
          )}
        </div>
      </div>

      {/* Template Suggestions */}
      <div className="space-y-2">
        <p className="text-xs text-zinc-500 uppercase tracking-wider">
          Quick prompts
        </p>
        <div className="flex flex-wrap gap-2">
          {PROMPT_TEMPLATES.map((t, i) => (
            <button
              key={i}
              onClick={() => setPrompt(t)}
              className="text-xs px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 rounded-full text-zinc-400 hover:text-white transition-colors truncate max-w-xs"
              title={t}
            >
              {t.slice(0, 60)}...
            </button>
          ))}
        </div>
      </div>

      {/* Error / Status */}
      {error && (
        <div className="p-4 bg-red-900/30 border border-red-700 rounded-lg text-red-300 text-sm">
          {error}
        </div>
      )}
      {status && (
        <div className="p-4 bg-emerald-900/30 border border-emerald-700 rounded-lg text-emerald-300 text-sm">
          {status}
        </div>
      )}

      {/* Generated Image Preview */}
      {generatedImage && (
        <div className="space-y-2">
          <p className="text-sm text-zinc-400">Preview:</p>
          <div className="border border-zinc-700 rounded-lg overflow-hidden">
            <img
              src={generatedImage}
              alt="Generated kitchen"
              className="w-full max-h-[600px] object-contain bg-zinc-900"
            />
          </div>
          <p className="text-xs text-zinc-600 break-all">{usedPrompt}</p>
        </div>
      )}
    </div>
  );
}
