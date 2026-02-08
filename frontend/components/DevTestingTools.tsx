"use client";

import { NormalizedAnalysisResult } from '@/types/analysis';

/**
 * DevTestingTools Component (DEV ONLY)
 * 
 * Provides mock analysis results for testing Solana minting without running the ML model.
 * Only visible when NODE_ENV === 'development'.
 * 
 * These tools help test:
 * - SolanaCertificatePanel with eligible REAL result (score >= 85)
 * - SolanaCertificatePanel with ineligible FAKE result (blocks minting)
 */

interface DevTestingToolsProps {
    onLoadResult: (result: NormalizedAnalysisResult) => void;
}

// Generate a small mock spectrogram (20x20 grid)
function generateMockSpectrogram(): number[][] {
    const rows = 20;
    const cols = 20;
    const spectrogram: number[][] = [];
    for (let i = 0; i < rows; i++) {
        const row: number[] = [];
        for (let j = 0; j < cols; j++) {
            // Create a wave-like pattern
            row.push(Math.sin(i * 0.3) * Math.cos(j * 0.3) * 0.5 + 0.5);
        }
        spectrogram.push(row);
    }
    return spectrogram;
}

export function DevTestingTools({ onLoadResult }: DevTestingToolsProps) {
    // Only render in development
    if (process.env.NODE_ENV !== 'development') {
        return null;
    }

    const loadEligibleResult = () => {
        const mockResult: NormalizedAnalysisResult = {
            overall_score: 95,
            verdict: 'REAL',
            explanation: 'Dev mock: This audio exhibits natural breathing patterns and organic prosody.',
            gemini_explanation: 'DEV TEST: Natural speech characteristics detected. This is a mock result for testing NFT minting eligibility.',
            audio_hash: `sha256:dev-test-${Date.now()}`,
            analysis: {
                breathing: 'natural',
                prosody_variation: 'high',
                frequency_spectrum: 'organic',
                speaking_rhythm: 'consistent'
            },
            artifacts: [],
            timeline_data: Array.from({ length: 20 }, (_, i) => ({
                time: i * 0.5,
                confidence: 0.9 + Math.random() * 0.1
            })),
            audio_fingerprint: {
                spectrogram: generateMockSpectrogram(),
                frequency_bins: [],
                amplitude_bins: [],
                time_bins: []
            },
            segments: [
                { start: 0, end: 1.5, score: 95 },
                { start: 1.5, end: 3.0, score: 93 },
                { start: 3.0, end: 4.5, score: 96 }
            ],
            spectrogram: generateMockSpectrogram()
        };
        onLoadResult(mockResult);
    };

    const loadFakeResult = () => {
        const mockResult: NormalizedAnalysisResult = {
            overall_score: 20,
            verdict: 'FAKE',
            explanation: 'Dev mock: This audio shows clear signs of AI generation.',
            gemini_explanation: 'DEV TEST: Synthetic audio detected. This is a mock result for testing NFT minting rejection.',
            audio_hash: `sha256:dev-fake-${Date.now()}`,
            analysis: {
                breathing: 'unnatural',
                prosody_variation: 'low',
                frequency_spectrum: 'mechanical',
                speaking_rhythm: 'irregular'
            },
            artifacts: [
                { timestamp: 0.5, type: 'glitch', description: 'Metallic artifact detected' },
                { timestamp: 2.1, type: 'discontinuity', description: 'Phase shift in audio' }
            ],
            timeline_data: Array.from({ length: 20 }, (_, i) => ({
                time: i * 0.5,
                confidence: 0.15 + Math.random() * 0.1
            })),
            audio_fingerprint: {
                spectrogram: generateMockSpectrogram(),
                frequency_bins: [],
                amplitude_bins: [],
                time_bins: []
            },
            segments: [
                { start: 0, end: 1.5, score: 18 },
                { start: 1.5, end: 3.0, score: 22 },
                { start: 3.0, end: 4.5, score: 19 }
            ],
            spectrogram: generateMockSpectrogram()
        };
        onLoadResult(mockResult);
    };

    return (
        <div className="fixed bottom-4 left-4 z-50 flex flex-col gap-2 p-3 bg-yellow-900/90 border border-yellow-500/50 rounded-lg shadow-lg">
            <div className="text-[10px] font-bold text-yellow-300 uppercase tracking-wider mb-1">
                🛠️ DEV TOOLS
            </div>
            <button
                onClick={loadEligibleResult}
                className="px-3 py-1.5 text-xs font-medium bg-green-600 hover:bg-green-700 text-white rounded transition-colors"
            >
                Load REAL Result (95%)
            </button>
            <button
                onClick={loadFakeResult}
                className="px-3 py-1.5 text-xs font-medium bg-red-600 hover:bg-red-700 text-white rounded transition-colors"
            >
                Load FAKE Result (20%)
            </button>
            <div className="text-[9px] text-yellow-400/70 mt-1">
                For testing NFT minting
            </div>
        </div>
    );
}
