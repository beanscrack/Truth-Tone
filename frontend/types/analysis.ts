/**
 * Normalized analysis result type used across all components.
 * This is the canonical format - all API responses should be normalized to this.
 */
export interface NormalizedAnalysisResult {
    overall_score: number; // 0-100 realness confidence
    verdict: 'REAL' | 'LIKELY REAL' | 'UNCERTAIN' | 'LIKELY FAKE' | 'FAKE';
    segments: Array<{ start: number; end: number; score: number }>; // score 0-100 realness
    spectrogram: number[][];
    frequencies?: number[];
    gemini_explanation: string;
    audio_hash: string;
    message?: string;
    mode?: string;
    // Legacy fields for backward compatibility with existing UI
    confidence_score?: number; // 0-1 (legacy)
    explanation?: string; // legacy alias
    analysis?: {
        breathing: string;
        prosody_variation: string;
        frequency_spectrum: string;
        speaking_rhythm: string;
    };
    artifacts?: Array<{
        timestamp: number;
        type: string;
        description: string;
    }>;
    timeline_data?: Array<{ time: number; confidence: number }>;
    audio_fingerprint?: {
        spectrogram?: number[][];
        frequency_bins: number[];
        amplitude_bins: number[];
        time_bins: number[];
    };
}

/**
 * Raw API response from /api/analyze - may be in either format
 */
export interface RawAnalysisResponse {
    // New format (0-100)
    overall_score?: number;
    segments?: Array<{ start: number; end: number; score: number }>;
    gemini_explanation?: string;
    audio_hash?: string;
    // Legacy format (0-1)
    confidence_score?: number;
    explanation?: string;
    // Common fields
    verdict: string;
    mode?: string;
    spectrogram?: number[][];
    frequencies?: number[];
    analysis?: {
        breathing: string;
        prosody_variation: string;
        frequency_spectrum: string;
        speaking_rhythm: string;
    };
    artifacts?: Array<{
        timestamp: number;
        type: string;
        description: string;
    }>;
    timeline_data?: Array<{ time: number; confidence: number }>;
    audio_fingerprint?: {
        spectrogram?: number[][];
        frequency_bins: number[];
        amplitude_bins: number[];
        time_bins: number[];
    };
}

/**
 * Normalize any API response to the canonical format.
 * Handles both legacy (0-1) and new (0-100) score formats.
 */
export function normalizeAnalysisResult(raw: RawAnalysisResponse): NormalizedAnalysisResult {
    // Determine overall_score (0-100)
    let overall_score: number;
    if (typeof raw.overall_score === 'number') {
        overall_score = raw.overall_score;
    } else if (typeof raw.confidence_score === 'number') {
        // Convert 0-1 to 0-100
        overall_score = raw.confidence_score * 100;
    } else {
        overall_score = 50; // Default fallback
    }

    // Normalize verdict
    const validVerdicts = ['REAL', 'LIKELY REAL', 'UNCERTAIN', 'LIKELY FAKE', 'FAKE'] as const;
    const verdict = validVerdicts.includes(raw.verdict as typeof validVerdicts[number])
        ? (raw.verdict as NormalizedAnalysisResult['verdict'])
        : 'UNCERTAIN';

    // Normalize segments - if missing, create from timeline_data
    let segments: NormalizedAnalysisResult['segments'] = [];
    if (raw.segments && Array.isArray(raw.segments)) {
        segments = raw.segments;
    } else if (raw.timeline_data && Array.isArray(raw.timeline_data)) {
        // Convert timeline_data to segments format
        segments = raw.timeline_data.map((item, index, arr) => ({
            start: item.time,
            end: arr[index + 1]?.time ?? item.time + 1,
            score: item.confidence * 100, // timeline_data uses 0-1 format
        }));
    }

    // Get spectrogram from either location
    const spectrogram = raw.spectrogram ?? raw.audio_fingerprint?.spectrogram ?? [];

    // Get explanation
    const gemini_explanation = raw.gemini_explanation ?? raw.explanation ?? '';

    // Generate audio_hash if missing
    const audio_hash = raw.audio_hash ?? `hash_${Date.now().toString(36)}`;

    return {
        overall_score,
        verdict,
        segments,
        spectrogram,
        frequencies: raw.frequencies,
        gemini_explanation,
        audio_hash,
        mode: raw.mode,
        // Keep legacy fields for backward compatibility
        confidence_score: overall_score / 100,
        explanation: gemini_explanation,
        analysis: raw.analysis,
        artifacts: raw.artifacts,
        timeline_data: raw.timeline_data,
        audio_fingerprint: raw.audio_fingerprint,
    };
}

/**
 * Check if a result is eligible for NFT minting.
 * Requirements: verdict is REAL or LIKELY REAL AND overall_score >= 85
 */
export function isEligibleForCertificate(result: NormalizedAnalysisResult): boolean {
    const eligibleVerdicts = ['REAL', 'LIKELY REAL'];
    return eligibleVerdicts.includes(result.verdict) && result.overall_score >= 85;
}
