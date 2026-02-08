"use client";

import { useMemo, useState, useCallback } from 'react';
import { normalizeScore, getScoreColorHSL, formatTime } from './normalize';

export interface Segment {
    start: number;
    end: number;
    score: number;
    reason?: string;
}

export interface TimelineHeatmapProps {
    segments: Segment[];
    durationSec: number;
    currentTimeSec?: number;
    onSeek?: (timeSec: number) => void;
    verdict?: string;
}

interface TooltipState {
    visible: boolean;
    x: number;
    y: number;
    segment: Segment | null;
}

export function TimelineHeatmap({
    segments,
    durationSec,
    currentTimeSec,
    onSeek,
    verdict,
}: TimelineHeatmapProps) {
    const [tooltip, setTooltip] = useState<TooltipState>({
        visible: false,
        x: 0,
        y: 0,
        segment: null,
    });

    // Normalize segments and ensure valid data
    const normalizedSegments = useMemo(() => {
        if (!segments || !Array.isArray(segments) || segments.length === 0) {
            // Create a single segment spanning full duration
            return [{
                start: 0,
                end: durationSec || 10,
                score: 50, // uncertain
                reason: 'No segment data available',
            }];
        }

        return segments.map(seg => ({
            ...seg,
            score: normalizeScore(seg.score),
        }));
    }, [segments, durationSec]);

    // Calculate effective duration
    const effectiveDuration = useMemo(() => {
        if (durationSec > 0) return durationSec;
        if (normalizedSegments.length > 0) {
            return Math.max(...normalizedSegments.map(s => s.end));
        }
        return 10; // fallback
    }, [durationSec, normalizedSegments]);

    // Handle mouse events
    const handleMouseEnter = useCallback((e: React.MouseEvent, segment: Segment) => {
        const rect = e.currentTarget.getBoundingClientRect();
        setTooltip({
            visible: true,
            x: e.clientX - rect.left,
            y: -40, // above the bar
            segment,
        });
    }, []);

    const handleMouseLeave = useCallback(() => {
        setTooltip(prev => ({ ...prev, visible: false }));
    }, []);

    const handleClick = useCallback((segment: Segment) => {
        if (onSeek) {
            const midpoint = (segment.start + segment.end) / 2;
            onSeek(midpoint);
        }
    }, [onSeek]);

    // Calculate playhead position
    const playheadPosition = useMemo(() => {
        if (currentTimeSec === undefined || currentTimeSec < 0) return null;
        return (currentTimeSec / effectiveDuration) * 100;
    }, [currentTimeSec, effectiveDuration]);

    return (
        <div className="relative w-full">
            {/* Label */}
            <div className="flex justify-between items-center mb-2 px-1">
                <span className="text-xs font-medium text-neutral-400">Timeline Analysis</span>
                <span className="text-[10px] text-neutral-500">
                    {formatTime(0)} — {formatTime(effectiveDuration)}
                </span>
            </div>

            {/* Heatmap Container */}
            <div className="relative h-8 bg-neutral-900/50 rounded-lg overflow-hidden border border-white/5">
                {/* Segments */}
                <div className="absolute inset-0 flex">
                    {normalizedSegments.map((segment, index) => {
                        const startPercent = (segment.start / effectiveDuration) * 100;
                        const widthPercent = ((segment.end - segment.start) / effectiveDuration) * 100;

                        return (
                            <div
                                key={index}
                                className="absolute h-full cursor-pointer transition-opacity hover:opacity-80"
                                style={{
                                    left: `${startPercent}%`,
                                    width: `${Math.max(widthPercent, 1)}%`, // min 1% visibility
                                    backgroundColor: getScoreColorHSL(segment.score),
                                }}
                                onMouseEnter={(e) => handleMouseEnter(e, segment)}
                                onMouseLeave={handleMouseLeave}
                                onClick={() => handleClick(segment)}
                            />
                        );
                    })}
                </div>

                {/* Playhead */}
                {playheadPosition !== null && (
                    <div
                        className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg z-10"
                        style={{ left: `${playheadPosition}%` }}
                    >
                        <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-white rounded-full" />
                    </div>
                )}
            </div>

            {/* Tooltip */}
            {tooltip.visible && tooltip.segment && (
                <div
                    className="absolute z-20 px-3 py-2 bg-neutral-800 border border-white/10 rounded-lg shadow-xl text-xs pointer-events-none"
                    style={{
                        left: `${(tooltip.segment.start / effectiveDuration) * 100}%`,
                        bottom: '100%',
                        marginBottom: '8px',
                        transform: 'translateX(-50%)',
                        minWidth: '160px',
                    }}
                >
                    <div className="flex justify-between items-center gap-4 mb-1">
                        <span className="text-neutral-400">Time</span>
                        <span className="text-white font-mono">
                            {formatTime(tooltip.segment.start)} – {formatTime(tooltip.segment.end)}
                        </span>
                    </div>
                    <div className="flex justify-between items-center gap-4 mb-1">
                        <span className="text-neutral-400">Score</span>
                        <span
                            className="font-bold"
                            style={{ color: getScoreColorHSL(tooltip.segment.score) }}
                        >
                            {tooltip.segment.score.toFixed(0)}%
                        </span>
                    </div>
                    {tooltip.segment.reason && (
                        <div className="text-neutral-400 border-t border-white/10 pt-1 mt-1">
                            {tooltip.segment.reason}
                        </div>
                    )}
                    {!tooltip.segment.reason && (
                        <div className="text-neutral-500 border-t border-white/10 pt-1 mt-1 italic">
                            Model flagged this region.
                        </div>
                    )}
                    {/* Arrow */}
                    <div className="absolute left-1/2 -bottom-1 -translate-x-1/2 w-2 h-2 bg-neutral-800 border-r border-b border-white/10 rotate-45" />
                </div>
            )}

            {/* Legend */}
            <div className="flex items-center justify-center gap-4 mt-2 text-[10px] text-neutral-500">
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: getScoreColorHSL(90) }} />
                    <span>Real</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: getScoreColorHSL(50) }} />
                    <span>Uncertain</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: getScoreColorHSL(20) }} />
                    <span>Fake</span>
                </div>
            </div>
        </div>
    );
}
