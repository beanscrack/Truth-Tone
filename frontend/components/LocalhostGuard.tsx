"use client";

import { useEffect, useState } from 'react';

/**
 * LocalhostGuard Component (DEV ONLY)
 * 
 * Shows a warning banner when the app is accessed via LAN IP instead of localhost.
 * 
 * WHY THIS EXISTS:
 * - Phantom and other wallet browser extensions only reliably inject on localhost
 * - LAN IPs (10.x.x.x, 192.168.x.x, etc.) often fail wallet injection
 * - This is a security feature of browser extensions, not a bug
 * 
 * This component only renders in development (NODE_ENV === 'development')
 * and automatically hides in production builds.
 */
export function LocalhostGuard() {
    const [showWarning, setShowWarning] = useState(false);
    const [currentHost, setCurrentHost] = useState('');

    useEffect(() => {
        // Only check in development
        if (process.env.NODE_ENV !== 'development') return;

        const hostname = window.location.hostname;
        setCurrentHost(hostname);

        // Check if NOT on localhost or 127.0.0.1
        const isLocalhost = hostname === 'localhost' || hostname === '127.0.0.1';
        setShowWarning(!isLocalhost);
    }, []);

    // Don't render anything in production or if on localhost
    if (process.env.NODE_ENV !== 'development' || !showWarning) {
        return null;
    }

    return (
        <div className="fixed top-0 left-0 right-0 z-50 bg-yellow-500/90 text-black px-4 py-2 text-center text-sm font-medium">
            <span className="mr-2">⚠️</span>
            Wallet extensions require <strong>localhost</strong>.
            You&apos;re on <code className="bg-yellow-600/30 px-1 rounded">{currentHost}</code>.
            <a
                href="http://localhost:3000"
                className="ml-2 underline font-bold hover:text-yellow-900"
            >
                Open localhost:3000 →
            </a>
        </div>
    );
}
