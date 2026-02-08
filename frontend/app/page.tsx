"use client";

import { useState, useRef, useCallback, useEffect } from 'react';
import axios from 'axios';
import { Upload, Mic, Play, ShieldCheck, ShieldAlert, Sparkles, Activity, FileAudio, ArrowRight } from 'lucide-react';
import { AudioFingerprint } from '@/components/viz/AudioFingerprint';
import { TimelineHeatmap } from '@/components/viz/TimelineHeatmap';
import { ConfidenceGauge } from '@/components/viz/ConfidenceGauge';
import { WalletButton } from '@/components/WalletButton';
import { SolanaCertificatePanel } from '@/components/SolanaCertificatePanel';
import { DevTestingTools } from '@/components/DevTestingTools';
import { normalizeAnalysisResult, NormalizedAnalysisResult } from '@/types/analysis';

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
    spectrogram?: number[][];
    frequency_bins: number[];
    amplitude_bins: number[];
    time_bins: number[];
  };
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<NormalizedAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [activeTab, setActiveTab] = useState<'upload' | 'generate'>('upload');
  const [textToGenerate, setTextToGenerate] = useState('');

  // Audio playback ref for TimelineHeatmap seek
  const audioRef = useRef<HTMLAudioElement>(null);

  // DEV: Audio source state (changes based on REAL vs FAKE test)
  const [devAudioSrc, setDevAudioSrc] = useState('/test-audio-real.mp3');
  const [devAudioLabel, setDevAudioLabel] = useState<string | null>(null);

  // Handle seek from TimelineHeatmap click
  const handleSeek = useCallback((timeSec: number) => {
    console.log(`[TimelineHeatmap] Seek to: ${timeSec.toFixed(2)}s`);
    if (audioRef.current) {
      audioRef.current.currentTime = timeSec;
      audioRef.current.play().catch(() => {
        // Autoplay might be blocked, that's okay
      });
    }
  }, []);

  // Force audio reload when source changes
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.load();
    }
  }, [devAudioSrc]);

  // DEV: Handle audio source change from test tools
  const handleSetAudioSrc = useCallback((src: string, label: string) => {
    console.log(`[DEV] Setting audio source: ${src} (${label})`);
    setDevAudioSrc(src);
    setDevAudioLabel(label);
    // Force audio element to reload with new source
    if (audioRef.current) {
      audioRef.current.load();
    }
  }, []);


  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setResult(null);
      setError(null);

      // Update audio player source to the uploaded file
      const objectUrl = URL.createObjectURL(selectedFile);
      setDevAudioSrc(objectUrl);
      setDevAudioLabel('UPLOADED');
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await axios.post('/api/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(normalizeAnalysisResult(response.data));
    } catch (err: any) {
      console.error(err);
      setError("Analysis failed. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleGenerate = async () => {
    if (!textToGenerate.trim()) return;
    setIsAnalyzing(true);
    setError(null);
    setResult(null);
    try {
      const response = await axios.post('/api/generate-fake', { text: textToGenerate });

      // Update audio player if filename is returned
      if (response.data.filename) {
        // We are statically serving the generated files (including fallback) at /files
        // Assuming default backend URL or env var
        const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        const audioUrl = `${backendUrl}/files/${response.data.filename}`;

        console.log("Setting generated audio src:", audioUrl);
        setDevAudioSrc(audioUrl);
        setDevAudioLabel('GENERATED AI AUDIO');
      }

      if (response.data.analysis) {
        const normalized = normalizeAnalysisResult(response.data.analysis);
        // Add message from backend response to the result object
        normalized.message = response.data.message;
        setResult(normalized);
      } else {
        setError("Generation succeeded but analysis data is missing.");
      }
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.error || "Generation failed. Ensure API key is set.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <main className="min-h-screen bg-[#050505] text-white font-sans selection:bg-white/20">

      {/* Navbar - Glassmorphism, minimal */}
      <nav className="fixed top-0 w-full z-50 border-b border-white/[0.06] bg-[#050505]/80 backdrop-blur-xl">
        <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-5 h-5 bg-white rounded-full flex items-center justify-center">
              <div className="w-2 h-2 bg-black rounded-full" />
            </div>
            <span className="text-sm font-medium tracking-tight text-white">TruthTone</span>
          </div>
          <div className="flex gap-4">
            <button className="text-xs font-medium text-white/70 hover:text-white transition-colors">Documentation</button>
            <button className="text-xs font-medium text-white/70 hover:text-white transition-colors">API</button>
            <WalletButton />
          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <div className="pt-32 pb-20 px-6 max-w-4xl mx-auto">

        {/* Hero Section - Centered, minimal typography */}
        <section className="text-center space-y-4 mb-16 animate-in fade-in slide-in-from-bottom-4 duration-700">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/[0.03] border border-white/[0.08] mb-4">
            <Sparkles className="w-3 h-3 text-cyan-400" suppressHydrationWarning />
            <span className="text-[10px] uppercase tracking-wider font-semibold text-neutral-400">Deepfake Detection v1.0</span>
          </div>
          <h1 className="text-4xl md:text-6xl font-semibold tracking-tight text-white">
            Uncover the <span className="text-neutral-500">invisible.</span>
          </h1>
          <p className="text-neutral-400 text-lg max-w-lg mx-auto leading-relaxed">
            Advanced audio forensics powered by multimodal AI. Distinguish authentic human speech from synthetic artifacts with precision.
          </p>
        </section>

        {/* Interaction Card - The "App" */}
        <div className="bg-[#0A0A0A] border border-white/[0.06] rounded-2xl overflow-hidden shadow-2xl shadow-black/50 animate-in fade-in zoom-in duration-500 delay-100">

          {/* Tab Navigation */}
          <div className="flex border-b border-white/[0.06]">
            <button
              onClick={() => {
                setActiveTab('upload');
                setResult(null);
                setError(null);
                setTextToGenerate('');
                setDevAudioLabel(null);
              }}
              className={`flex-1 py-4 text-sm font-medium transition-all flex items-center justify-center gap-2 ${activeTab === 'upload' ? 'text-white bg-white/[0.02]' : 'text-white/50 hover:text-white hover:bg-white/[0.01]'}`}
            >
              <FileAudio className="w-4 h-4" suppressHydrationWarning />
              Analyze File
            </button>
            <div className="w-[1px] bg-white/[0.06]" />
            <button
              onClick={() => {
                setActiveTab('generate');
                setResult(null);
                setError(null);
                setFile(null);
                setDevAudioLabel(null);
              }}
              className={`flex-1 py-4 text-sm font-medium transition-all flex items-center justify-center gap-2 ${activeTab === 'generate' ? 'text-white bg-white/[0.02]' : 'text-white/50 hover:text-white hover:bg-white/[0.01]'}`}
            >
              <Mic className="w-4 h-4" suppressHydrationWarning />
              Test Generation
            </button>
          </div>

          {/* Content Body */}
          <div className="p-8 md:p-12 min-h-[400px] flex flex-col justify-center">

            {/* Mode: Upload */}
            {activeTab === 'upload' && !result && (
              <div className="flex flex-col items-center gap-6">
                {!file ? (
                  <label className="group cursor-pointer flex flex-col items-center gap-4 p-12 w-full border border-dashed border-white/10 rounded-xl hover:bg-white/[0.02] hover:border-white/20 transition-all">
                    <div className="w-12 h-12 bg-white/[0.03] rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
                      <Upload className="w-5 h-5 text-white/60 group-hover:text-white transition-colors" suppressHydrationWarning />
                    </div>
                    <div className="text-center">
                      <p className="text-sm font-medium text-white mb-1">Click to upload or drag and drop</p>
                      <p className="text-xs text-neutral-500">WAV, MP3, M4A up to 10MB</p>
                    </div>
                    <input type="file" className="hidden" onChange={handleFileChange} accept="audio/*" />
                  </label>
                ) : (
                  <div className="w-full space-y-6">
                    <div className="flex items-center justify-between p-4 bg-white/[0.03] rounded-lg border border-white/[0.06]">
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 bg-white/[0.05] rounded-full flex items-center justify-center">
                          <Play className="w-4 h-4 text-white" />
                        </div>
                        <div>
                          <p className="text-sm font-medium text-white">{file.name}</p>
                          <p className="text-xs text-neutral-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                        </div>
                      </div>
                      {!isAnalyzing && (
                        <button onClick={() => setFile(null)} className="text-xs text-neutral-500 hover:text-red-400 transition-colors">
                          Remove
                        </button>
                      )}
                    </div>
                    <button
                      onClick={handleAnalyze}
                      disabled={isAnalyzing}
                      className="w-full py-3 bg-white text-black text-sm font-semibold rounded-lg hover:bg-neutral-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                      {isAnalyzing ? (
                        <>
                          <div className="w-4 h-4 border-2 border-black/30 border-t-black rounded-full animate-spin" />
                          Processing...
                        </>
                      ) : (
                        <>Run Identification <ArrowRight className="w-4 h-4" /></>
                      )}
                    </button>
                  </div>
                )}
              </div>
            )}


            {/* Mode: Generate */}
            {activeTab === 'generate' && !result && (
              <div className="space-y-6 w-full max-w-lg mx-auto">
                <div className="space-y-2">
                  <label className="text-xs font-semibold text-neutral-400 uppercase tracking-wider">Prompt</label>
                  <textarea
                    value={textToGenerate}
                    onChange={(e) => setTextToGenerate(e.target.value)}
                    placeholder="Enter text to synthesize..."
                    className="w-full h-32 bg-[#050505] border border-white/[0.1] rounded-lg p-4 text-sm text-neutral-200 focus:outline-none focus:border-white/20 transition-all resize-none placeholder:text-neutral-700"
                  />
                </div>
                <button
                  onClick={handleGenerate}
                  disabled={isAnalyzing || !textToGenerate.trim()}
                  className="w-full py-3 bg-white text-black text-sm font-semibold rounded-lg hover:bg-neutral-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isAnalyzing ? (
                    <>
                      <div className="w-4 h-4 border-2 border-black/30 border-t-black rounded-full animate-spin" />
                      Synthesizing...
                    </>
                  ) : (
                    <>Generate & Identify <Sparkles className="w-4 h-4" /></>
                  )}
                </button>
              </div>
            )}

            {/* Result View */}
            {result && (
              <div className="space-y-8 animate-in fade-in slide-in-from-bottom-2 duration-500">
                {/* Status Header with Confidence Gauge */}
                <div className="flex items-center justify-between pb-6 border-b border-white/[0.06]">
                  <div className="flex items-center gap-4">
                    <div className={`w-12 h-12 rounded-full flex items-center justify-center ${['REAL', 'LIKELY REAL'].includes(result.verdict) ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}`}>
                      {['REAL', 'LIKELY REAL'].includes(result.verdict) ? <ShieldCheck className="w-6 h-6" /> : <ShieldAlert className="w-6 h-6" />}
                    </div>
                    <div>
                      <h2 className="text-lg font-semibold text-white">{result.verdict}</h2>
                      <p className="text-sm text-neutral-500">Confidence Score</p>
                    </div>
                  </div>
                  <div className="flex justify-center">
                    <ConfidenceGauge score={result.overall_score} verdict={result.verdict} size={120} />
                  </div>
                </div>

                {/* Explanation */}
                <div className="space-y-2">
                  <h3 className="text-xs font-semibold text-neutral-500 uppercase tracking-widest">Analysis Insight</h3>
                  <p className="text-sm text-neutral-300 leading-relaxed bg-white/[0.02] p-4 rounded-lg border border-white/[0.06]">
                    {result.gemini_explanation || result.explanation || 'No explanation available.'}
                  </p>
                </div>

                {/* Grid Stats */}
                {result.analysis && (
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                    {Object.entries(result.analysis).map(([key, value]) => (
                      <div key={key} className="p-3 bg-white/[0.02] border border-white/[0.06] rounded-lg">
                        <p className="text-[10px] text-neutral-500 uppercase tracking-wider mb-1">{key.replace('_', ' ')}</p>
                        <p className="text-sm font-medium text-white capitalize">{value}</p>
                      </div>
                    ))}
                  </div>
                )}

                {/* Timeline Heatmap */}
                {result.segments && result.segments.length > 0 && (
                  <div className="p-4 bg-white/[0.02] border border-white/[0.06] rounded-lg space-y-3">
                    <TimelineHeatmap
                      segments={result.segments}
                      durationSec={result.segments[result.segments.length - 1]?.end || 10}
                      verdict={result.verdict}
                      onSeek={handleSeek}
                    />
                    {/* Audio player for seek testing */}
                    <div className="pt-2 border-t border-white/[0.06]">
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-[10px] text-neutral-500">🎧 Click a heatmap segment to jump playback</p>
                        {devAudioLabel && (
                          <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${devAudioLabel === 'REAL'
                            ? 'bg-green-500/20 text-green-400'
                            : 'bg-red-500/20 text-red-400'
                            }`}>
                            DEV AUDIO: {devAudioLabel}
                          </span>
                        )}
                      </div>
                      <audio
                        ref={audioRef}
                        controls
                        className="w-full h-8 opacity-80"
                        src={devAudioSrc}
                      >
                        Your browser does not support audio playback.
                      </audio>
                    </div>
                  </div>
                )}

                {/* 3D Audio Fingerprint Visualization */}
                <div className="h-72 rounded-lg overflow-hidden border border-white/[0.06] relative group">
                  <AudioFingerprint
                    spectrogram={result.audio_fingerprint?.spectrogram || result.spectrogram || []}
                    verdict={result.verdict}
                    overallScore={result.overall_score}
                  />
                </div>

                {/* Actions */}
                <div className="space-y-4">
                  {/* Generation Status Message */}
                  {result.message && (
                    <div className={`p-3 rounded-lg text-sm border ${result.message.includes('simulated')
                      ? 'bg-yellow-500/10 border-yellow-500/20 text-yellow-400'
                      : 'bg-green-500/10 border-green-500/20 text-green-400'
                      }`}>
                      <div className="flex items-center gap-2">
                        {result.message.includes('simulated') ? <ShieldAlert className="w-4 h-4" /> : <ShieldCheck className="w-4 h-4" />}
                        <span>{result.message}</span>
                      </div>
                      {result.message.includes('simulated') && (
                        <div className="mt-1 text-xs opacity-80 pl-6">
                          The backend used a pre-generated file because the ElevenLabs API Key is missing or invalid. Check your .env file.
                        </div>
                      )}
                    </div>
                  )}

                  <button onClick={() => { setResult(null); setFile(null); }} className="w-full py-3 text-sm font-medium text-neutral-500 hover:text-white transition-colors border border-white/[0.06] rounded-lg hover:bg-white/[0.02]">
                    Start New Analysis
                  </button>
                </div>

                {/* NFT Certificate Panel */}
                <div className="mt-4">
                  <SolanaCertificatePanel analysisResult={result} />
                </div>
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="mt-6 p-4 bg-red-500/5 border border-red-500/10 rounded-lg flex items-center gap-3">
                <Activity className="w-4 h-4 text-red-400" />
                <span className="text-sm text-red-400">{error}</span>
              </div>
            )}

          </div>
        </div>

        {/* Footer */}
        <footer className="mt-20 text-center text-xs text-neutral-600">
          <p>TruthTone AI • 2026</p>
        </footer>

      </div>

      {/* DEV Testing Tools - only visible in development */}
      <DevTestingTools onLoadResult={setResult} onSetAudioSrc={handleSetAudioSrc} />
    </main>
  );
}
