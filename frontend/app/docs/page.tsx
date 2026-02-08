"use client";

import Link from 'next/link';
import { BookOpen, ShieldCheck, Cpu, Mic, GitBranch, ArrowLeft } from 'lucide-react';

export default function DocumentationPage() {
    return (
        <main className="min-h-screen bg-[#050505] text-white font-sans selection:bg-cyan-500/30">

            {/* Navbar */}
            <nav className="fixed top-0 w-full z-50 border-b border-white/[0.06] bg-[#050505]/80 backdrop-blur-xl">
                <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
                    <Link href="/" className="flex items-center gap-2 text-white/70 hover:text-white transition-colors">
                        <ArrowLeft className="w-4 h-4" />
                        <span className="text-sm font-medium">Back to App</span>
                    </Link>
                    <div className="flex items-center gap-3">
                        <span className="text-sm font-semibold tracking-tight text-white">TruthTone Docs</span>
                    </div>
                </div>
            </nav>

            <div className="pt-32 pb-20 px-6 max-w-4xl mx-auto space-y-16">

                {/* Header */}
                <section className="space-y-4">
                    <h1 className="text-4xl font-bold tracking-tight text-white">Documentation</h1>
                    <p className="text-lg text-neutral-400 max-w-2xl">
                        Learn how TruthTone uses advanced multimodal AI to detect deepfake audio with high precision.
                    </p>
                </section>

                {/* Core Concepts */}
                <section className="space-y-8">
                    <h2 className="text-2xl font-semibold text-white flex items-center gap-3">
                        <BookOpen className="w-5 h-5 text-cyan-400" />
                        Core Concepts
                    </h2>

                    <div className="grid md:grid-cols-2 gap-6">
                        <div className="p-6 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                            <div className="w-10 h-10 rounded-lg bg-cyan-500/10 flex items-center justify-center mb-4">
                                <ShieldCheck className="w-5 h-5 text-cyan-400" />
                            </div>
                            <h3 className="text-lg font-medium text-white mb-2">Hybrid Analysis</h3>
                            <p className="text-sm text-neutral-400 leading-relaxed">
                                TruthTone combines traditional signal processing (Librosa) with Large Language Model reasoning (Gemini 2.0 Flash) to analyze audio artifacts that are invisible to the naked ear.
                            </p>
                        </div>

                        <div className="p-6 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                            <div className="w-10 h-10 rounded-lg bg-purple-500/10 flex items-center justify-center mb-4">
                                <Cpu className="w-5 h-5 text-purple-400" />
                            </div>
                            <h3 className="text-lg font-medium text-white mb-2">Spectrogram Analysis</h3>
                            <p className="text-sm text-neutral-400 leading-relaxed">
                                Our ML connection converts audio into Mel Spectrograms, treating them as images to detect visual artifacts in the frequency domain that are characteristic of synthetic generation.
                            </p>
                        </div>
                    </div>
                </section>

                {/* How it Works */}
                <section className="space-y-8">
                    <h2 className="text-2xl font-semibold text-white flex items-center gap-3">
                        <GitBranch className="w-5 h-5 text-green-400" />
                        Detection Pipeline
                    </h2>

                    <div className="space-y-4 relative pl-8 border-l border-white/[0.1]">
                        <div className="absolute left-[-5px] top-0 w-2.5 h-2.5 rounded-full bg-cyan-500 box-content border-4 border-[#050505]" />
                        <div className="space-y-1">
                            <h3 className="text-base font-medium text-white">1. Input Processing</h3>
                            <p className="text-sm text-neutral-400">Audio is normalized and converted to 16kHz mono WAV format.</p>
                        </div>
                    </div>

                    <div className="space-y-4 relative pl-8 border-l border-white/[0.1]">
                        <div className="absolute left-[-5px] top-0 w-2.5 h-2.5 rounded-full bg-blue-500 box-content border-4 border-[#050505]" />
                        <div className="space-y-1">
                            <h3 className="text-base font-medium text-white">2. Feature Extraction</h3>
                            <p className="text-sm text-neutral-400">We extract Mel Spectrograms, spectral roll-off, and flatness metrics.</p>
                        </div>
                    </div>

                    <div className="space-y-4 relative pl-8 border-l border-white/[0.1]">
                        <div className="absolute left-[-5px] top-0 w-2.5 h-2.5 rounded-full bg-purple-500 box-content border-4 border-[#050505]" />
                        <div className="space-y-1">
                            <h3 className="text-base font-medium text-white">3. AI Inference</h3>
                            <p className="text-sm text-neutral-400">
                                Gemini 2.0 Flash analyzes the audio context for semantic anomalies (e.g., unnatural pauses, lack of breaths) while our ResNet-based classifier looks for synthesis artifacts.
                            </p>
                        </div>
                    </div>

                    <div className="space-y-4 relative pl-8 border-l-0">
                        <div className="absolute left-[-5px] top-0 w-2.5 h-2.5 rounded-full bg-green-500 box-content border-4 border-[#050505]" />
                        <div className="space-y-1">
                            <h3 className="text-base font-medium text-white">4. Certification</h3>
                            <p className="text-sm text-neutral-400">
                                Results are hashed and can be minted as a Solana NFT to provide immutable proof of authenticity.
                            </p>
                        </div>
                    </div>

                </section>

            </div>
        </main>
    );
}
