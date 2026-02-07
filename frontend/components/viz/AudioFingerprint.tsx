"use client";

import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

interface AudioFingerprintProps {
    data: number[][]; // 2D array of spectrogram data
}

function Terrain({ data }: { data: number[][] }) {
    const meshRef = useRef<THREE.Mesh>(null);

    // Generate geometry from data
    const geometry = useMemo(() => {
        if (!data || data.length === 0) return new THREE.PlaneGeometry(10, 10, 10, 10);

        const width = data.length; // Frequency bins
        const height = data[0].length; // Time steps

        const geo = new THREE.PlaneGeometry(10, 10, width - 1, height - 1);

        // Modify vertex heights based on data
        const positionAttribute = geo.getAttribute('position');

        for (let i = 0; i < positionAttribute.count; i++) {
            // Map 1D index to 2D grid
            const x = i % width;
            const y = Math.floor(i / width);

            // Safety check
            if (x < width && y < height) {
                const value = data[x][y]; // Normalized 0-1
                positionAttribute.setZ(i, value * 2); // Height multiplier
            }
        }

        geo.computeVertexNormals();
        return geo;
    }, [data]);

    useFrame((state) => {
        if (meshRef.current) {
            meshRef.current.rotation.z += 0.001; // Slow rotation
        }
    });

    return (
        <mesh ref={meshRef} geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
            <meshStandardMaterial
                color="#06b6d4"
                wireframe={true}
                side={THREE.DoubleSide}
                emissive="#06b6d4"
                emissiveIntensity={0.5}
            />
        </mesh>
    );
}

export function AudioFingerprint({ data }: AudioFingerprintProps) {
    return (
        <div className="w-full h-full">
            <Canvas>
                <PerspectiveCamera makeDefault position={[0, 5, 10]} />
                <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={0.5} />
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} />

                <Terrain data={data} />

                <gridHelper args={[20, 20, 0xffffff, 0x222222]} position={[0, -1, 0]} />
            </Canvas>
        </div>
    );
}
