"use client";

import Link from 'next/link';
import { Network, Code, Server, ArrowLeft } from 'lucide-react';

export default function ApiDocsPage() {
    return (
        <main className="min-h-screen bg-[#050505] text-white font-sans selection:bg-purple-500/30">

            {/* Navbar */}
            <nav className="fixed top-0 w-full z-50 border-b border-white/[0.06] bg-[#050505]/80 backdrop-blur-xl">
                <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
                    <Link href="/" className="flex items-center gap-2 text-white/70 hover:text-white transition-colors">
                        <ArrowLeft className="w-4 h-4" />
                        <span className="text-sm font-medium">Back to App</span>
                    </Link>
                    <div className="flex items-center gap-3">
                        <span className="text-sm font-semibold tracking-tight text-white">TruthTone API</span>
                    </div>
                </div>
            </nav>

            <div className="pt-32 pb-20 px-6 max-w-4xl mx-auto space-y-16">

                {/* Header */}
                <section className="space-y-4">
                    <h1 className="text-4xl font-bold tracking-tight text-white">API Reference</h1>
                    <p className="text-lg text-neutral-400 max-w-2xl">
                        Integrate TruthTone deepfake detection into your own applications using our RESTful API.
                    </p>
                </section>

                {/* POST /analyze */}
                <section className="space-y-8 pb-10 border-b border-white/[0.06]">
                    <div className="flex items-center gap-4">
                        <div className="bg-green-500/10 text-green-400 px-3 py-1 rounded text-xs font-mono font-bold uppercase tracking-wider border border-green-500/20">POST</div>
                        <h2 className="text-2xl font-mono text-white tracking-tight">/api/analyze</h2>
                    </div>

                    <p className="text-neutral-400">
                        Submit an audio file for deepfake analysis. Supports standard audio formats (MP3, WAV, M4A) up to 10MB.
                    </p>

                    <div className="bg-[#0A0A0A] border border-white/[0.06] rounded-xl overflow-hidden">
                        <div className="flex items-center justify-between px-4 py-3 bg-white/[0.02] border-b border-white/[0.06]">
                            <span className="text-xs font-mono text-neutral-500">Request Body (Multipart)</span>
                            <Code className="w-4 h-4 text-neutral-600" />
                        </div>
                        <div className="p-4 font-mono text-sm text-neutral-300 space-y-2">
                            <div className="flex gap-8">
                                <span className="w-24 text-purple-400">file</span>
                                <span className="text-neutral-500">Binary</span>
                                <span className="text-white/60">The audio file to analyze (required)</span>
                            </div>
                        </div>
                    </div>

                    <div className="bg-[#0A0A0A] border border-white/[0.06] rounded-xl overflow-hidden mt-6">
                        <div className="flex items-center justify-between px-4 py-3 bg-white/[0.02] border-b border-white/[0.06]">
                            <span className="text-xs font-mono text-neutral-500">Response (JSON)</span>
                            <Server className="w-4 h-4 text-neutral-600" />
                        </div>
                        <pre className="p-4 font-mono text-xs text-neutral-300 overflow-x-auto">
                            {`{
  "confidence_score": 0.98,
  "verdict": "REAL",
  "explanation": "Natural prosody and consistent background noise...",
  "analysis": {
    "breathing": "Natural intakes present",
    "prosody_variation": "High dynamic range",
    "frequency_spectrum": "Organic falloff > 16kHz",
    "speaking_rhythm": "Irregular (human-like)"
  },
  "artifacts": [],
  "timeline_data": [ ... ],
  "audio_fingerprint": { ... }
}`}
                        </pre>
                    </div>
                </section>

                {/* POST /generate-fake */}
                <section className="space-y-8">
                    <div className="flex items-center gap-4">
                        <div className="bg-green-500/10 text-green-400 px-3 py-1 rounded text-xs font-mono font-bold uppercase tracking-wider border border-green-500/20">POST</div>
                        <h2 className="text-2xl font-mono text-white tracking-tight">/api/generate-fake</h2>
                    </div>
                    <p className="text-neutral-400">
                        Generate synthetic speech for testing purposes using ElevenLabs (or fallback simulation).
                    </p>

                    <div className="bg-[#0A0A0A] border border-white/[0.06] rounded-xl overflow-hidden">
                        <div className="flex items-center justify-between px-4 py-3 bg-white/[0.02] border-b border-white/[0.06]">
                            <span className="text-xs font-mono text-neutral-500">Request Body (JSON)</span>
                            <Code className="w-4 h-4 text-neutral-600" />
                        </div>
                        <div className="p-4 font-mono text-sm text-neutral-300 space-y-2">
                            <div className="flex gap-8">
                                <span className="w-24 text-purple-400">text</span>
                                <span className="text-neutral-500">String</span>
                                <span className="text-white/60">The text content to synthesize (required)</span>
                            </div>
                        </div>
                    </div>
                </section>

            </div>
        </main>
    );
}
