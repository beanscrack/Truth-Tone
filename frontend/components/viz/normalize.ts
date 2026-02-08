/**
 * Visualization Normalization Utilities
 * 
 * Handles score normalization for visualization components.
 * Backend may return scores in range 0..1 OR 0..100.
 */

/**
 * Normalize a score to 0-100 range
 * - If score <= 1, assumes 0..1 range and converts to percentage
 * - If score > 1, assumes already in 0..100 range
 */
export function normalizeScore(score: number): number {
    if (score === undefined || score === null || isNaN(score)) {
        return 50; // default uncertain
    }
    // If score is in 0..1 range, convert to percentage
    if (score <= 1) {
        return score * 100;
    }
    // Clamp to 0-100
    return Math.max(0, Math.min(100, score));
}

/**
 * Get color for a score on the green-yellow-red scale
 * Higher score = more REAL = greener
 * Lower score = more FAKE = redder
 */
export function getScoreColor(score: number): string {
    const normalized = normalizeScore(score);

    if (normalized >= 85) {
        return '#22c55e'; // green-500 (REAL)
    } else if (normalized >= 70) {
        return '#84cc16'; // lime-500 (LIKELY REAL)
    } else if (normalized >= 50) {
        return '#eab308'; // yellow-500 (UNCERTAIN)
    } else if (normalized >= 30) {
        return '#f97316'; // orange-500 (LIKELY FAKE)
    } else {
        return '#ef4444'; // red-500 (FAKE)
    }
}

/**
 * Get HSL color for smooth gradient
 * Score 0 = red (0°), Score 100 = green (120°)
 */
export function getScoreColorHSL(score: number): string {
    const normalized = normalizeScore(score);
    // Map 0-100 to hue 0-120 (red to green)
    const hue = (normalized / 100) * 120;
    return `hsl(${hue}, 70%, 50%)`;
}

/**
 * Format time in mm:ss format
 */
export function formatTime(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Determine if a verdict indicates the audio is real
 */
export function isRealVerdict(verdict: string): boolean {
    return ['REAL', 'LIKELY REAL'].includes(verdict?.toUpperCase() ?? '');
}

/**
 * Determine if a verdict indicates the audio is fake
 */
export function isFakeVerdict(verdict: string): boolean {
    return ['FAKE', 'LIKELY FAKE'].includes(verdict?.toUpperCase() ?? '');
}
