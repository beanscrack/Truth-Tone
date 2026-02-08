"use client";

import { useState, useEffect } from 'react';
import { useWallet } from '@/providers/WalletProvider';

/**
 * WalletButton Component
 * 
 * Renders a button for Phantom wallet connection.
 * Uses a mounted state to prevent hydration mismatches caused by:
 * - Browser extensions (Dark Reader, etc.) modifying SVG attributes
 * - Server/client state differences for wallet detection
 */
export function WalletButton() {
    const { connected, publicKey, phantomInstalled, connecting, connect, disconnect } = useWallet();
    const [mounted, setMounted] = useState(false);

    // Only render dynamic content after mount to prevent hydration mismatch
    useEffect(() => {
        setMounted(true);
    }, []);

    // Format address to show first 4 and last 4 characters
    const formatAddress = (address: string) => {
        return `${address.slice(0, 4)}...${address.slice(-4)}`;
    };

    // Render a stable placeholder during SSR to avoid hydration mismatch
    // This prevents Dark Reader and similar extensions from causing issues
    if (!mounted) {
        return (
            <button
                className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-neutral-400 bg-neutral-800 border border-neutral-700 rounded-lg"
                disabled
            >
                <span className="w-3.5 h-3.5 bg-neutral-600 rounded-sm" />
                Wallet
            </button>
        );
    }

    // Not installed - show install button
    if (!phantomInstalled) {
        return (
            <button
                onClick={() => window.open('https://phantom.app/', '_blank')}
                className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-purple-400 bg-purple-500/10 border border-purple-500/20 rounded-lg hover:bg-purple-500/20 transition-colors"
            >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
                </svg>
                Install Phantom
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
            </button>
        );
    }

    // Connected - show address and disconnect
    if (connected && publicKey) {
        return (
            <div className="flex items-center gap-2">
                <div className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-green-400 bg-green-500/10 border border-green-500/20 rounded-lg">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                    <span title={publicKey}>{formatAddress(publicKey)}</span>
                    <span className="text-green-600 text-[10px]">devnet</span>
                </div>
                <button
                    onClick={disconnect}
                    className="p-1.5 text-neutral-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
                    title="Disconnect wallet"
                >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                    </svg>
                </button>
            </div>
        );
    }

    // Not connected - show connect button
    return (
        <button
            onClick={connect}
            disabled={connecting}
            className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-white bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
            {connecting ? (
                <>
                    <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Connecting...
                </>
            ) : (
                <>
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
                    </svg>
                    Connect Phantom
                </>
            )}
        </button>
    );
}
