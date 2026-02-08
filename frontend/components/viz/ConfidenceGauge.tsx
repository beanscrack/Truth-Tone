"use client";

import { useEffect, useState, useMemo } from 'react';
import { normalizeScore, getScoreColorHSL, getScoreColor } from './normalize';

export interface ConfidenceGaugeProps {
    score: number;
    verdict?: string;
    size?: number; // diameter in pixels
    strokeWidth?: number;
    animationDuration?: number; // ms
}

export function ConfidenceGauge({
    score,
    verdict,
    size = 180,
    strokeWidth = 12,
    animationDuration = 1200,
}: ConfidenceGaugeProps) {
    const [animatedScore, setAnimatedScore] = useState(0);

    // Normalize the score to 0-100
    const normalizedScore = useMemo(() => normalizeScore(score), [score]);

    // Animate the score from 0 to target
    useEffect(() => {
        const startTime = Date.now();
        const startValue = animatedScore;
        const targetValue = normalizedScore;

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / animationDuration, 1);

            // Spring-like easing (ease out back)
            const eased = 1 - Math.pow(1 - progress, 3);

            const currentValue = startValue + (targetValue - startValue) * eased;
            setAnimatedScore(currentValue);

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        requestAnimationFrame(animate);
    }, [normalizedScore, animationDuration]);

    // Calculate SVG parameters
    const radius = (size - strokeWidth) / 2;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference * (1 - animatedScore / 100);

    // Color based on score
    const gaugeColor = getScoreColorHSL(animatedScore);
    const solidColor = getScoreColor(Math.round(animatedScore));

    // Determine verdict text color
    const getVerdictColor = (v: string) => {
        const upper = v?.toUpperCase() ?? '';
        if (upper === 'REAL') return 'text-green-400';
        if (upper === 'LIKELY REAL') return 'text-lime-400';
        if (upper === 'UNCERTAIN') return 'text-yellow-400';
        if (upper === 'LIKELY FAKE') return 'text-orange-400';
        if (upper === 'FAKE') return 'text-red-400';
        return 'text-neutral-400';
    };

    return (
        <div className="relative inline-flex flex-col items-center">
            <svg
                width={size}
                height={size}
                viewBox={`0 0 ${size} ${size}`}
                className="transform -rotate-90"
            >
                {/* Background circle */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke="rgba(255, 255, 255, 0.1)"
                    strokeWidth={strokeWidth}
                />

                {/* Animated progress arc */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke={gaugeColor}
                    strokeWidth={strokeWidth}
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    strokeDashoffset={strokeDashoffset}
                    className="transition-colors duration-300"
                    style={{
                        filter: `drop-shadow(0 0 8px ${solidColor}40)`,
                    }}
                />

                {/* Glow effect */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke={gaugeColor}
                    strokeWidth={strokeWidth * 0.3}
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    strokeDashoffset={strokeDashoffset}
                    opacity={0.5}
                    style={{
                        filter: 'blur(4px)',
                    }}
                />
            </svg>

            {/* Center content */}
            <div className="absolute inset-0 flex flex-col items-center justify-center">
                {/* Score number */}
                <span
                    className="text-4xl font-bold transition-colors duration-300"
                    style={{ color: solidColor }}
                >
                    {Math.round(animatedScore)}%
                </span>

                {/* Verdict label */}
                {verdict && (
                    <span className={`text-xs font-semibold mt-1 ${getVerdictColor(verdict)}`}>
                        {verdict}
                    </span>
                )}

                {/* Small label */}
                <span className="text-[10px] text-neutral-500 mt-1">
                    Confidence
                </span>
            </div>

            {/* Ticks */}
            {[0, 25, 50, 75, 100].map((tick) => {
                const angle = (tick / 100) * 360 - 90;
                const tickRadius = radius + strokeWidth / 2 + 8;
                const x = size / 2 + tickRadius * Math.cos((angle * Math.PI) / 180);
                const y = size / 2 + tickRadius * Math.sin((angle * Math.PI) / 180);

                return (
                    <span
                        key={tick}
                        className="absolute text-[8px] text-neutral-600 font-mono"
                        style={{
                            left: x,
                            top: y,
                            transform: 'translate(-50%, -50%)',
                        }}
                    >
                        {tick}
                    </span>
                );
            })}
        </div>
    );
}
