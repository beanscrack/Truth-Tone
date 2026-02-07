"use client";

import { useState } from 'react';
import axios from 'axios';
import { Upload, Mic, Play, ShieldCheck, ShieldAlert } from 'lucide-react';
import { AudioFingerprint } from '@/components/viz/AudioFingerprint';
// import { TimelineHeatmap } from '@/components/viz/TimelineHeatmap';
// import { VerificationCertificate } from '@/components/Certificate';

interface AnalysisResult {
  confidence_score: number;
  verdict: string;
  explanation: string;
  analysis: {
    breathing: string;
    prosody_variation: string;
    frequency_spectrum: string;
    speaking_rhythm: string;
  };
  artifacts: Array<{
    timestamp: number;
    type: string;
    description: string;
  }>;
  timeline_data: Array<{ time: number, confidence: number }>;
  audio_fingerprint: {
    frequency_bins: number[];
    amplitude_bins: number[];
    time_bins: number[];
  };
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Call our Next.js API proxy
      const response = await axios.post('/api/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
    } catch (err: any) {
      console.error(err);
      setError("Analysis failed. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <main className="min-h-screen bg-neutral-950 text-white font-sans selection:bg-cyan-500/30">
      {/* Header */}
      <header className="fixed top-0 w-full z-50 border-b border-white/10 bg-neutral-950/80 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-lg animate-pulse" />
            <span className="text-xl font-bold tracking-tight">TruthTone<span className="text-cyan-400">++</span></span>
          </div>
          <button className="px-4 py-2 bg-white/5 hover:bg-white/10 rounded-full text-sm font-medium transition-colors border border-white/5">
            Connect Wallet
          </button>
        </div>
      </header>

      <div className="pt-32 pb-20 px-6 max-w-5xl mx-auto space-y-12">

        {/* Hero Section */}
        <section className="text-center space-y-6">
          <h1 className="text-5xl md:text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-white to-white/40 pb-2">
            Is it Real or AI?
          </h1>
          <p className="text-lg text-neutral-400 max-w-2xl mx-auto">
            Advanced deepfake detection powered by Gemini + Audio Signal Analysis.
          </p>
        </section>

        {/* Upload Container */}
        <div className="relative group">
          <div className="absolute -inset-1 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-1000"></div>
          <div className="relative bg-neutral-900 border border-white/10 rounded-xl p-10 flex flex-col items-center justify-center gap-6 min-h-[300px]">

            {!file && !result ? (
              <>
                <div className="w-20 h-20 bg-neutral-800 rounded-full flex items-center justify-center mb-2">
                  <Upload className="w-8 h-8 text-neutral-400" />
                </div>
                <div className="text-center space-y-2">
                  <label htmlFor="audio-upload" className="cursor-pointer">
                    <span className="bg-white text-black px-6 py-3 rounded-full font-semibold hover:scale-105 transition-transform inline-block">
                      Upload Audio
                    </span>
                    <input
                      id="audio-upload"
                      type="file"
                      accept="audio/*"
                      className="hidden"
                      onChange={handleFileChange}
                    />
                  </label>
                  <p className="text-sm text-neutral-500">Supports WAV, MP3, M4A</p>
                </div>
              </>
            ) : (
              <div className="w-full space-y-6 animate-in fade-in zoom-in duration-300">
                {/* File Preview */}
                <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg border border-white/10">
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 bg-cyan-500/20 rounded-full flex items-center justify-center">
                      <Play className="w-5 h-5 text-cyan-400" />
                    </div>
                    <div>
                      <p className="font-medium text-white">{file?.name}</p>
                      <p className="text-xs text-neutral-400">{(file?.size! / 1024 / 1024).toFixed(2)} MB</p>
                    </div>
                  </div>
                  {!result && !isAnalyzing && (
                    <button
                      onClick={() => setFile(null)}
                      className="text-xs text-red-400 hover:text-red-300"
                    >
                      Remove
                    </button>
                  )}
                </div>

                {!result && (
                  <button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className="w-full py-4 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-lg font-bold text-lg hover:shadow-[0_0_20px_rgba(6,182,212,0.5)] transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {isAnalyzing ? (
                      <>
                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        Analyzing Signal...
                      </>
                    ) : (
                      "Run Deepfake Detection"
                    )}
                  </button>
                )}
              </div>
            )}

            {/* Error State */}
            {error && (
              <div className="w-full p-4 bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg text-sm text-center">
                {error}
              </div>
            )}

          </div>
        </div>

        {/* Results Section */}
        {result && (
          <div className="space-y-8 animate-in slide-in-from-bottom-10 duration-700">

            {/* Main Verdict Card */}
            <div className={`p-8 rounded-2xl border ${result.verdict === 'REAL' ? 'bg-green-500/10 border-green-500/20' : 'bg-red-500/10 border-red-500/20'}`}>
              <div className="flex flex-col md:flex-row items-center justify-between gap-6">
                <div className="space-y-2">
                  <h2 className="text-sm font-medium text-neutral-400 uppercase tracking-widest">AI Verdict</h2>
                  <div className={`text-5xl font-bold flex items-center gap-3 ${result.verdict === 'REAL' ? 'text-green-400' : 'text-red-400'}`}>
                    {result.verdict === 'REAL' ? <ShieldCheck className="w-12 h-12" /> : <ShieldAlert className="w-12 h-12" />}
                    {result.verdict}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-6xl font-black text-white">
                    {(result.confidence_score * 100).toFixed(1)}<span className="text-2xl text-neutral-500">%</span>
                  </div>
                  <p className="text-sm text-neutral-400">Confidence Score</p>
                </div>
              </div>

              {/* Gemini Explanation */}
              <div className="mt-8 pt-8 border-t border-white/10 space-y-6">

                {/* Text Explanation */}
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 bg-purple-500/20 rounded flex items-center justify-center shrink-0">
                    <span className="text-purple-400 text-xs font-bold">AI</span>
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-white mb-1">Generated Insight</h3>
                    <p className="text-neutral-300 leading-relaxed text-sm">
                      {result.explanation}
                    </p>
                  </div>
                </div>

                {/* Analysis Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(result.analysis).map(([key, value]) => (
                    <div key={key} className="bg-white/5 p-4 rounded-lg border border-white/5">
                      <p className="text-xs text-neutral-500 uppercase tracking-wider mb-1">{key.replace('_', ' ')}</p>
                      <p className="text-sm font-medium text-cyan-300 capitalize">{value}</p>
                    </div>
                  ))}
                </div>

              </div>
            </div>

            {/* Visualization Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Placeholder for 3D Viz */}
              <div className="bg-neutral-900 border border-white/10 rounded-xl p-6 h-[400px] flex items-center justify-center relative overflow-hidden group">
                <div className="absolute inset-0 bg-grid-white/[0.02] -z-10" />
                {/* Mapped raw freq data to mimic 2D spectrogram input for now */}
                <AudioFingerprint data={[result.audio_fingerprint.frequency_bins.map(v => v * 0.5)]} />
              </div>

              {/* Placeholder for Heatmap/Timeline */}
              <div className="bg-neutral-900 border border-white/10 rounded-xl p-6 h-[400px] flex items-center justify-center relative">
                <p className="text-neutral-500 font-mono text-sm">Timeline Heatmap Placeholder</p>
                {/* <TimelineHeatmap segments={result.timeline_data} /> */}
              </div>
            </div>

          </div>
        )}

      </div>
    </main>
  );
}
