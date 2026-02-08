"use client";

import { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import { isRealVerdict, isFakeVerdict } from './normalize';

export interface AudioFingerprintProps {
    spectrogram?: number[][];
    data?: number[][]; // legacy alias
    verdict?: string;
    overallScore?: number;
}

interface TerrainProps {
    data: number[][];
    isReal: boolean;
    isFake: boolean;
    score: number;
}

function Terrain({ data, isReal, isFake, score }: TerrainProps) {
    const meshRef = useRef<THREE.Mesh>(null);
    const originalPositions = useRef<Float32Array | null>(null);
    const timeRef = useRef(0);

    // Generate geometry from data
    const geometry = useMemo(() => {
        // Safe check: data exists, is an array, and has inner arrays (2D)
        if (!data || !Array.isArray(data) || data.length === 0 || !Array.isArray(data[0])) {
            // Return a flat plane as placeholder
            return new THREE.PlaneGeometry(10, 10, 20, 20);
        }

        const width = Math.min(data.length, 50); // Cap resolution for performance
        const height = Math.min(data[0].length, 50);

        const geo = new THREE.PlaneGeometry(10, 10, width - 1, height - 1);
        const positionAttribute = geo.getAttribute('position');

        for (let i = 0; i < positionAttribute.count; i++) {
            const x = i % width;
            const y = Math.floor(i / width);

            if (x < width && y < height) {
                let value = data[x % data.length][y % data[0].length] || 0;

                // Normalize value to 0-1 if needed
                if (value > 1) value = value / 100;

                // Apply different surface characteristics based on verdict
                if (isReal) {
                    // REAL: More organic, irregular jagged surface
                    const noise = Math.sin(x * 0.7) * Math.cos(y * 0.5) * 0.15;
                    const irregularity = Math.random() * 0.1 - 0.05;
                    value = value * (1 + noise + irregularity);
                } else if (isFake) {
                    // FAKE: Smoother, more symmetric, uncanny
                    const smoothFactor = Math.sin(x * 0.3) * Math.sin(y * 0.3) * 0.3;
                    value = value * 0.8 + smoothFactor * 0.2; // Blend toward smooth
                }

                positionAttribute.setZ(i, value * 2.5); // Height multiplier
            }
        }

        geo.computeVertexNormals();
        return geo;
    }, [data, isReal, isFake]);

    // Store original positions for animation
    useEffect(() => {
        if (geometry) {
            const positions = geometry.getAttribute('position').array;
            originalPositions.current = new Float32Array(positions.length);
            originalPositions.current.set(positions);
        }
    }, [geometry]);

    // Animate vertices based on verdict
    useFrame((state, delta) => {
        if (!meshRef.current || !originalPositions.current) return;

        timeRef.current += delta;
        const time = timeRef.current;

        const positionAttribute = meshRef.current.geometry.getAttribute('position');
        const positions = positionAttribute.array as Float32Array;
        const width = Math.ceil(Math.sqrt(positionAttribute.count));

        for (let i = 0; i < positionAttribute.count; i++) {
            const x = i % width;
            const y = Math.floor(i / width);
            const originalZ = originalPositions.current[i * 3 + 2];

            let displacement = 0;

            if (isReal) {
                // REAL: Subtle chaotic organic motion
                displacement =
                    Math.sin(time * 1.3 + x * 0.4 + y * 0.3) * 0.03 +
                    Math.cos(time * 0.9 + x * 0.2 - y * 0.5) * 0.02 +
                    Math.sin(time * 2.1 + x * 0.6 + y * 0.8) * 0.015;
            } else if (isFake) {
                // FAKE: Periodic harmonic wave (uncanny, mechanical)
                displacement =
                    Math.sin(time * 2.0 + x * 0.2) * 0.04 +
                    Math.sin(time * 2.0 + y * 0.2) * 0.04;
            } else {
                // Neutral: Very subtle ambient motion
                displacement = Math.sin(time + i * 0.01) * 0.01;
            }

            positions[i * 3 + 2] = originalZ + displacement;
        }

        positionAttribute.needsUpdate = true;

        // Slow rotation
        meshRef.current.rotation.z += 0.001;
    });

    // Color based on verdict
    const meshColor = useMemo(() => {
        if (isReal) return '#22c55e'; // green
        if (isFake) return '#ef4444'; // red
        return '#06b6d4'; // cyan (neutral)
    }, [isReal, isFake]);

    const emissiveIntensity = useMemo(() => {
        if (isReal) return 0.4;
        if (isFake) return 0.3;
        return 0.5;
    }, [isReal, isFake]);

    return (
        <mesh ref={meshRef} geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
            <meshStandardMaterial
                color={meshColor}
                wireframe={true}
                side={THREE.DoubleSide}
                emissive={meshColor}
                emissiveIntensity={emissiveIntensity}
            />
        </mesh>
    );
}

export function AudioFingerprint({
    spectrogram,
    data,
    verdict = '',
    overallScore = 50
}: AudioFingerprintProps) {
    // Support both prop names
    const spectrogramData = spectrogram || data || [];

    // Determine visualization mode
    const isReal = isRealVerdict(verdict);
    const isFake = isFakeVerdict(verdict);

    // Normalize score
    const normalizedScore = overallScore <= 1 ? overallScore * 100 : overallScore;

    return (
        <div className="relative w-full h-full">
            <Canvas>
                <PerspectiveCamera makeDefault position={[0, 6, 10]} />
                <OrbitControls
                    enableZoom={false}
                    autoRotate
                    autoRotateSpeed={isReal ? 0.3 : isFake ? 0.8 : 0.5}
                    enablePan={false}
                    maxPolarAngle={Math.PI / 2.2}
                    minPolarAngle={Math.PI / 4}
                />
                <ambientLight intensity={0.4} />
                <pointLight position={[10, 10, 10]} intensity={0.8} />
                <pointLight position={[-10, 5, -10]} intensity={0.3} color="#06b6d4" />

                <Terrain
                    data={spectrogramData}
                    isReal={isReal}
                    isFake={isFake}
                    score={normalizedScore}
                />

                <gridHelper args={[20, 20, 0xffffff, 0x222222]} position={[0, -1, 0]} />
            </Canvas>

            {/* Verdict Badge */}
            {verdict && (
                <div className="absolute top-3 right-3 px-2 py-1 rounded text-xs font-semibold"
                    style={{
                        backgroundColor: isReal ? 'rgba(34, 197, 94, 0.2)' :
                            isFake ? 'rgba(239, 68, 68, 0.2)' :
                                'rgba(6, 182, 212, 0.2)',
                        color: isReal ? '#22c55e' :
                            isFake ? '#ef4444' :
                                '#06b6d4',
                        border: `1px solid ${isReal ? 'rgba(34, 197, 94, 0.3)' :
                            isFake ? 'rgba(239, 68, 68, 0.3)' :
                                'rgba(6, 182, 212, 0.3)'}`,
                    }}
                >
                    {verdict}
                </div>
            )}

            {/* Legend */}
            <div className="absolute bottom-3 left-3 text-[10px] text-neutral-500">
                <div className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: isReal ? '#22c55e' : isFake ? '#ef4444' : '#06b6d4' }}
                    />
                    <span>Spectral Fingerprint</span>
                </div>
            </div>
        </div>
    );
}
