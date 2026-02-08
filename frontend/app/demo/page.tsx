"use client";

import { useState, useRef, useCallback } from 'react';
import axios from 'axios';
import Link from 'next/link';
import {
    Mic,
    MicOff,
    Play,
    Pause,
    AlertCircle,
    Loader2,
    ArrowLeft,
    ShieldCheck,
    ShieldAlert,
    RefreshCw
} from 'lucide-react';
import { normalizeAnalysisResult, NormalizedAnalysisResult } from '@/types/analysis';

type DemoStep = 'idle' | 'recording' | 'generating' | 'ready' | 'analyzing' | 'complete';

interface AudioPair {
    original: { blob: Blob; url: string } | null;
    fake: { blob: Blob | null; url: string } | null;
}

interface ComparisonResult {
    original: NormalizedAnalysisResult | null;
    fake: NormalizedAnalysisResult | null;
}

export default function DemoPage() {
    const [step, setStep] = useState<DemoStep>('idle');
    const [error, setError] = useState<string | null>(null);
    const [countdown, setCountdown] = useState(10);
    const [audioPair, setAudioPair] = useState<AudioPair>({ original: null, fake: null });
    const [comparison, setComparison] = useState<ComparisonResult>({ original: null, fake: null });
    const [isPlayingOriginal, setIsPlayingOriginal] = useState(false);
    const [isPlayingFake, setIsPlayingFake] = useState(false);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const originalAudioRef = useRef<HTMLAudioElement | null>(null);
    const fakeAudioRef = useRef<HTMLAudioElement | null>(null);
    const countdownIntervalRef = useRef<NodeJS.Timeout | null>(null);

    // Reset error after 5 seconds
    const showError = useCallback((message: string) => {
        setError(message);
        setTimeout(() => setError(null), 5000);
    }, []);

    // Start recording
    const startRecording = async () => {
        setError(null);
        audioChunksRef.current = [];

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                const audioUrl = URL.createObjectURL(audioBlob);
                setAudioPair(prev => ({ ...prev, original: { blob: audioBlob, url: audioUrl } }));

                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());

                // Move to generating step
                generateFake(audioBlob);
            };

            // Start recording
            mediaRecorder.start();
            setStep('recording');
            setCountdown(10);

            // Countdown timer
            countdownIntervalRef.current = setInterval(() => {
                setCountdown(prev => {
                    if (prev <= 1) {
                        // Stop recording
                        if (mediaRecorderRef.current?.state === 'recording') {
                            mediaRecorderRef.current.stop();
                        }
                        if (countdownIntervalRef.current) {
                            clearInterval(countdownIntervalRef.current);
                        }
                        return 0;
                    }
                    return prev - 1;
                });
            }, 1000);

        } catch (err) {
            console.error('Microphone error:', err);
            if (err instanceof DOMException && err.name === 'NotAllowedError') {
                showError('Microphone permission denied. Please allow microphone access and try again.');
            } else {
                showError('Failed to access microphone. Please check your device settings.');
            }
            setStep('idle');
        }
    };

    // Stop recording early
    const stopRecording = () => {
        if (mediaRecorderRef.current?.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
        if (countdownIntervalRef.current) {
            clearInterval(countdownIntervalRef.current);
        }
    };

    // Generate fake audio from recorded audio
    const generateFake = async (originalBlob: Blob) => {
        setStep('generating');

        try {
            // For demo, we'll send a text prompt derived from "recorded audio"
            // In a full implementation, you might want voice cloning with the actual audio
            const response = await axios.post('/api/generate-fake', {
                text: 'This is a demonstration of AI-generated speech that sounds similar to human voice.'
            }, { timeout: 60000 });

            let fakeUrl: string;
            let fakeBlob: Blob | null = null;

            // Handle both URL and base64 responses
            if (response.data.audio_url) {
                fakeUrl = response.data.audio_url;
                // Try to fetch and create blob for analysis
                try {
                    const audioResponse = await fetch(fakeUrl);
                    fakeBlob = await audioResponse.blob();
                } catch {
                    // If fetch fails, we'll analyze using URL later
                }
            } else if (response.data.audio_base64) {
                // Convert base64 to blob
                const binaryString = atob(response.data.audio_base64);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                fakeBlob = new Blob([bytes], { type: 'audio/mp3' });
                fakeUrl = URL.createObjectURL(fakeBlob);
            } else {
                throw new Error('Invalid response: no audio data received');
            }

            setAudioPair(prev => ({ ...prev, fake: { blob: fakeBlob, url: fakeUrl } }));
            setStep('ready');

        } catch (err) {
            console.error('Generation error:', err);
            if (axios.isAxiosError(err) && err.code === 'ECONNABORTED') {
                showError('Request timed out. The server might be busy.');
            } else {
                showError('Failed to generate fake audio. Please try again.');
            }
            setStep('idle');
        }
    };

    // Run analysis on both audio files
    const runComparison = async () => {
        if (!audioPair.original || !audioPair.fake) return;

        setStep('analyzing');
        setComparison({ original: null, fake: null });

        try {
            // Analyze original
            const originalFormData = new FormData();
            originalFormData.append('file', audioPair.original.blob, 'original.webm');

            const originalResponse = await axios.post('/api/analyze', originalFormData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                timeout: 60000,
            });

            const originalResult = normalizeAnalysisResult(originalResponse.data);
            setComparison(prev => ({ ...prev, original: originalResult }));

            // Analyze fake
            if (audioPair.fake.blob) {
                const fakeFormData = new FormData();
                fakeFormData.append('file', audioPair.fake.blob, 'fake.mp3');

                const fakeResponse = await axios.post('/api/analyze', fakeFormData, {
                    headers: { 'Content-Type': 'multipart/form-data' },
                    timeout: 60000,
                });

                const fakeResult = normalizeAnalysisResult(fakeResponse.data);
                setComparison(prev => ({ ...prev, fake: fakeResult }));
            }

            setStep('complete');

        } catch (err) {
            console.error('Analysis error:', err);
            if (axios.isAxiosError(err) && err.code === 'ECONNABORTED') {
                showError('Analysis timed out. The audio might be too long.');
            } else {
                showError('Analysis failed. Please try again.');
            }
            setStep('ready');
        }
    };

    // Reset everything
    const reset = () => {
        // Revoke object URLs
        if (audioPair.original?.url) URL.revokeObjectURL(audioPair.original.url);
        if (audioPair.fake?.url && audioPair.fake.blob) URL.revokeObjectURL(audioPair.fake.url);

        setStep('idle');
        setAudioPair({ original: null, fake: null });
        setComparison({ original: null, fake: null });
        setCountdown(10);
        setError(null);
    };

    // Audio playback controls
    const toggleOriginal = () => {
        if (!originalAudioRef.current) return;
        if (isPlayingOriginal) {
            originalAudioRef.current.pause();
        } else {
            fakeAudioRef.current?.pause();
            setIsPlayingFake(false);
            originalAudioRef.current.play();
        }
        setIsPlayingOriginal(!isPlayingOriginal);
    };

    const toggleFake = () => {
        if (!fakeAudioRef.current) return;
        if (isPlayingFake) {
            fakeAudioRef.current.pause();
        } else {
            originalAudioRef.current?.pause();
            setIsPlayingOriginal(false);
            fakeAudioRef.current.play();
        }
        setIsPlayingFake(!isPlayingFake);
    };

    // Render score badge
    const ScoreBadge = ({ result }: { result: NormalizedAnalysisResult }) => {
        const isReal = result.verdict === 'REAL' || result.verdict === 'LIKELY REAL';
        return (
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium ${isReal ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                }`}>
                {isReal ? <ShieldCheck className="w-4 h-4" /> : <ShieldAlert className="w-4 h-4" />}
                {result.verdict} ({result.overall_score.toFixed(0)}%)
            </div>
        );
    };

    return (
        <main className="min-h-screen bg-[#050505] text-white font-sans">
            {/* Hidden audio elements */}
            {audioPair.original && (
                <audio
                    ref={originalAudioRef}
                    src={audioPair.original.url}
                    onEnded={() => setIsPlayingOriginal(false)}
                />
            )}
            {audioPair.fake && (
                <audio
                    ref={fakeAudioRef}
                    src={audioPair.fake.url}
                    onEnded={() => setIsPlayingFake(false)}
                />
            )}

            {/* Navbar */}
            <nav className="fixed top-0 w-full z-50 border-b border-white/[0.06] bg-[#050505]/80 backdrop-blur-xl">
                <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
                    <Link href="/" className="flex items-center gap-3 group">
                        <ArrowLeft className="w-4 h-4 text-neutral-500 group-hover:text-white transition-colors" />
                        <div className="flex items-center gap-2">
                            <div className="w-5 h-5 bg-white rounded-full flex items-center justify-center">
                                <div className="w-2 h-2 bg-black rounded-full" />
                            </div>
                            <span className="text-sm font-medium tracking-tight text-white">TruthTone Demo</span>
                        </div>
                    </Link>
                </div>
            </nav>

            {/* Main Content */}
            <div className="pt-32 pb-20 px-6 max-w-4xl mx-auto">
                {/* Header */}
                <section className="text-center space-y-4 mb-12">
                    <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-white">
                        Real vs Fake <span className="text-neutral-500">Demo</span>
                    </h1>
                    <p className="text-neutral-400 max-w-lg mx-auto">
                        Record your voice, generate an AI clone, and compare the analysis results side by side.
                    </p>
                </section>

                {/* Error Toast */}
                {error && (
                    <div className="mb-8 p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-3 animate-in fade-in slide-in-from-top-2">
                        <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
                        <p className="text-sm text-red-400">{error}</p>
                    </div>
                )}

                {/* Demo Card */}
                <div className="bg-[#0A0A0A] border border-white/[0.06] rounded-2xl overflow-hidden shadow-2xl shadow-black/50">
                    <div className="p-8 md:p-12">

                        {/* Step: Idle */}
                        {step === 'idle' && (
                            <div className="flex flex-col items-center gap-6 py-12">
                                <div className="w-20 h-20 bg-white/[0.03] rounded-full flex items-center justify-center">
                                    <Mic className="w-8 h-8 text-white/60" />
                                </div>
                                <div className="text-center">
                                    <h2 className="text-lg font-medium text-white mb-2">Step 1: Record Your Voice</h2>
                                    <p className="text-sm text-neutral-500">Click below to record 10 seconds of audio</p>
                                </div>
                                <button
                                    onClick={startRecording}
                                    className="px-6 py-3 bg-white text-black text-sm font-semibold rounded-lg hover:bg-neutral-200 transition-colors flex items-center gap-2"
                                >
                                    <Mic className="w-4 h-4" />
                                    Start Recording
                                </button>
                            </div>
                        )}

                        {/* Step: Recording */}
                        {step === 'recording' && (
                            <div className="flex flex-col items-center gap-6 py-12">
                                <div className="relative">
                                    <div className="w-24 h-24 bg-red-500/20 rounded-full flex items-center justify-center animate-pulse">
                                        <Mic className="w-10 h-10 text-red-400" />
                                    </div>
                                    <div className="absolute -inset-2 border-2 border-red-500/50 rounded-full animate-ping" />
                                </div>
                                <div className="text-center">
                                    <h2 className="text-lg font-medium text-white mb-2">Recording...</h2>
                                    <p className="text-4xl font-mono font-bold text-red-400">{countdown}s</p>
                                </div>
                                <button
                                    onClick={stopRecording}
                                    className="px-6 py-3 bg-red-600 text-white text-sm font-semibold rounded-lg hover:bg-red-700 transition-colors flex items-center gap-2"
                                >
                                    <MicOff className="w-4 h-4" />
                                    Stop Early
                                </button>
                            </div>
                        )}

                        {/* Step: Generating */}
                        {step === 'generating' && (
                            <div className="flex flex-col items-center gap-6 py-12">
                                <div className="w-20 h-20 bg-purple-500/20 rounded-full flex items-center justify-center">
                                    <Loader2 className="w-8 h-8 text-purple-400 animate-spin" />
                                </div>
                                <div className="text-center">
                                    <h2 className="text-lg font-medium text-white mb-2">Generating Fake Audio...</h2>
                                    <p className="text-sm text-neutral-500">Using AI to create a synthetic voice clone</p>
                                </div>
                            </div>
                        )}

                        {/* Step: Ready / Analyzing / Complete */}
                        {(step === 'ready' || step === 'analyzing' || step === 'complete') && (
                            <div className="space-y-8">
                                {/* Audio Players */}
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    {/* Original */}
                                    <div className="p-4 bg-white/[0.02] border border-white/[0.06] rounded-lg">
                                        <p className="text-xs font-semibold text-green-400 uppercase tracking-wider mb-3">Original (Your Voice)</p>
                                        <button
                                            onClick={toggleOriginal}
                                            disabled={!audioPair.original}
                                            className="w-full py-3 bg-green-500/10 hover:bg-green-500/20 border border-green-500/20 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
                                        >
                                            {isPlayingOriginal ? <Pause className="w-4 h-4 text-green-400" /> : <Play className="w-4 h-4 text-green-400" />}
                                            <span className="text-sm font-medium text-green-400">{isPlayingOriginal ? 'Pause' : 'Play'}</span>
                                        </button>
                                        {comparison.original && (
                                            <div className="mt-3">
                                                <ScoreBadge result={comparison.original} />
                                            </div>
                                        )}
                                    </div>

                                    {/* Fake */}
                                    <div className="p-4 bg-white/[0.02] border border-white/[0.06] rounded-lg">
                                        <p className="text-xs font-semibold text-red-400 uppercase tracking-wider mb-3">AI Generated (Fake)</p>
                                        <button
                                            onClick={toggleFake}
                                            disabled={!audioPair.fake}
                                            className="w-full py-3 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
                                        >
                                            {isPlayingFake ? <Pause className="w-4 h-4 text-red-400" /> : <Play className="w-4 h-4 text-red-400" />}
                                            <span className="text-sm font-medium text-red-400">{isPlayingFake ? 'Pause' : 'Play'}</span>
                                        </button>
                                        {comparison.fake && (
                                            <div className="mt-3">
                                                <ScoreBadge result={comparison.fake} />
                                            </div>
                                        )}
                                    </div>
                                </div>

                                {/* Compare Button */}
                                {step === 'ready' && (
                                    <button
                                        onClick={runComparison}
                                        className="w-full py-3 bg-white text-black text-sm font-semibold rounded-lg hover:bg-neutral-200 transition-colors"
                                    >
                                        Compare Both
                                    </button>
                                )}

                                {/* Analyzing State */}
                                {step === 'analyzing' && (
                                    <div className="flex items-center justify-center gap-3 py-4">
                                        <Loader2 className="w-5 h-5 text-purple-400 animate-spin" />
                                        <span className="text-sm text-purple-400">Analyzing both audio files...</span>
                                    </div>
                                )}

                                {/* Comparison Results */}
                                {step === 'complete' && comparison.original && comparison.fake && (
                                    <div className="space-y-4">
                                        <h3 className="text-sm font-semibold text-neutral-400 uppercase tracking-wider">Analysis Comparison</h3>

                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                            {/* Original Details */}
                                            <div className="p-4 bg-green-500/5 border border-green-500/10 rounded-lg">
                                                <p className="text-xs font-semibold text-green-400 mb-3">Original Audio</p>
                                                <div className="space-y-2 text-sm">
                                                    <div className="flex justify-between">
                                                        <span className="text-neutral-500">Verdict</span>
                                                        <span className="text-white font-medium">{comparison.original.verdict}</span>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <span className="text-neutral-500">Confidence</span>
                                                        <span className="text-white font-medium">{comparison.original.overall_score.toFixed(1)}%</span>
                                                    </div>
                                                </div>
                                                {comparison.original.gemini_explanation && (
                                                    <p className="mt-3 text-xs text-neutral-400 leading-relaxed line-clamp-3">
                                                        {comparison.original.gemini_explanation}
                                                    </p>
                                                )}
                                            </div>

                                            {/* Fake Details */}
                                            <div className="p-4 bg-red-500/5 border border-red-500/10 rounded-lg">
                                                <p className="text-xs font-semibold text-red-400 mb-3">AI Generated Audio</p>
                                                <div className="space-y-2 text-sm">
                                                    <div className="flex justify-between">
                                                        <span className="text-neutral-500">Verdict</span>
                                                        <span className="text-white font-medium">{comparison.fake.verdict}</span>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <span className="text-neutral-500">Confidence</span>
                                                        <span className="text-white font-medium">{comparison.fake.overall_score.toFixed(1)}%</span>
                                                    </div>
                                                </div>
                                                {comparison.fake.gemini_explanation && (
                                                    <p className="mt-3 text-xs text-neutral-400 leading-relaxed line-clamp-3">
                                                        {comparison.fake.gemini_explanation}
                                                    </p>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* Reset Button */}
                                <button
                                    onClick={reset}
                                    className="w-full py-3 text-sm font-medium text-neutral-500 hover:text-white transition-colors border border-white/[0.06] rounded-lg hover:bg-white/[0.02] flex items-center justify-center gap-2"
                                >
                                    <RefreshCw className="w-4 h-4" />
                                    Start Over
                                </button>
                            </div>
                        )}
                    </div>
                </div>

                {/* Help Text */}
                <p className="mt-8 text-center text-xs text-neutral-600">
                    This demo uses your microphone to record audio and compares it to AI-generated speech.
                    All processing happens locally and on our secure servers.
                </p>
            </div>
        </main>
    );
}
