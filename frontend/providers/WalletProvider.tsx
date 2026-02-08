"use client";

import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';

/**
 * ================================================================================
 * SOLANA WALLET CONFIGURATION NOTES
 * ================================================================================
 * 
 * NETWORK: The wallet connection itself is network-agnostic. The network (devnet,
 * mainnet) is determined when signing transactions, not during connect().
 * 
 * WHAT'S NEEDED FOR WALLET CONNECTION:
 * - Phantom browser extension installed (client-side only)
 * - No backend secrets or API keys required
 * - No wallet address sent to backend
 * 
 * ENVIRONMENT VARIABLES (frontend only):
 * - NEXT_PUBLIC_APP_URL: Used for NFT metadata URIs
 * - ML_API_URL: Backend URL (has nothing to do with wallet)
 * - NEXT_PUBLIC_DEMO_SIGNATURE: Optional demo fallback
 * 
 * THE BACKEND DOES NOT NEED:
 * - User's wallet address
 * - Any Solana RPC configuration
 * - Any wallet-related secrets
 * 
 * The only Solana interaction happens client-side when minting NFTs.
 * See utils/solana.ts for devnet configuration.
 * ================================================================================
 */

// Phantom wallet types
interface PhantomProvider {
    isPhantom?: boolean;
    publicKey?: { toString(): string; toBase58(): string };
    connect(opts?: { onlyIfTrusted?: boolean }): Promise<{ publicKey: { toString(): string; toBase58(): string } }>;
    disconnect(): Promise<void>;
    on(event: string, callback: (args: unknown) => void): void;
    off(event: string, callback: (args: unknown) => void): void;
}

interface WalletContextType {
    connected: boolean;
    publicKey: string | null;
    phantomInstalled: boolean;
    connecting: boolean;
    error: string | null;
    connect: () => Promise<void>;
    disconnect: () => void;
}

const WalletContext = createContext<WalletContextType>({
    connected: false,
    publicKey: null,
    phantomInstalled: false,
    connecting: false,
    error: null,
    connect: async () => { },
    disconnect: () => { },
});

export const useWallet = () => useContext(WalletContext);

/**
 * Get Phantom provider from window.solana
 * Only call this on the client side (inside useEffect or event handlers)
 */
const getPhantom = (): PhantomProvider | null => {
    if (typeof window === 'undefined') return null;

    // Phantom injects as window.solana or window.phantom.solana
    const windowAny = window as unknown as {
        solana?: PhantomProvider;
        phantom?: { solana?: PhantomProvider };
    };

    // Try window.phantom.solana first (newer injection method)
    if (windowAny.phantom?.solana?.isPhantom) {
        return windowAny.phantom.solana;
    }

    // Fallback to window.solana
    if (windowAny.solana?.isPhantom) {
        return windowAny.solana;
    }

    return null;
};

interface WalletProviderProps {
    children: ReactNode;
}

export function WalletProvider({ children }: WalletProviderProps) {
    const [connected, setConnected] = useState(false);
    const [publicKey, setPublicKey] = useState<string | null>(null);
    const [phantomInstalled, setPhantomInstalled] = useState(false);
    const [connecting, setConnecting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [mounted, setMounted] = useState(false);

    // Mark as mounted (client-side only)
    useEffect(() => {
        setMounted(true);
    }, []);

    // Check for Phantom after mount with retry logic
    useEffect(() => {
        if (!mounted) return;

        let attempts = 0;
        const maxAttempts = 10;
        const checkInterval = 100; // ms

        const checkPhantom = () => {
            const phantom = getPhantom();

            if (phantom) {
                setPhantomInstalled(true);

                // Check if already connected (trusted connection)
                if (phantom.publicKey) {
                    setConnected(true);
                    setPublicKey(phantom.publicKey.toBase58?.() || phantom.publicKey.toString());
                }
                return true;
            }

            attempts++;
            if (attempts < maxAttempts) {
                setTimeout(checkPhantom, checkInterval);
                return false;
            }

            // After max attempts, Phantom is not installed
            setPhantomInstalled(false);
            return false;
        };

        // Start checking
        checkPhantom();
    }, [mounted]);

    // Listen for account changes and disconnects
    useEffect(() => {
        if (!mounted || !phantomInstalled) return;

        const phantom = getPhantom();
        if (!phantom) return;

        const handleAccountChange = (newPublicKey: unknown) => {
            if (newPublicKey && typeof (newPublicKey as { toBase58?: () => string }).toBase58 === 'function') {
                const pk = (newPublicKey as { toBase58: () => string }).toBase58();
                setPublicKey(pk);
                setConnected(true);
            } else if (newPublicKey && typeof (newPublicKey as { toString?: () => string }).toString === 'function') {
                setPublicKey((newPublicKey as { toString: () => string }).toString());
                setConnected(true);
            } else {
                setPublicKey(null);
                setConnected(false);
            }
        };

        const handleDisconnect = () => {
            setPublicKey(null);
            setConnected(false);
        };

        phantom.on('accountChanged', handleAccountChange);
        phantom.on('disconnect', handleDisconnect);

        return () => {
            phantom.off('accountChanged', handleAccountChange);
            phantom.off('disconnect', handleDisconnect);
        };
    }, [mounted, phantomInstalled]);

    const connect = useCallback(async () => {
        setError(null);

        const phantom = getPhantom();
        if (!phantom) {
            // Open Phantom download page
            window.open('https://phantom.app/', '_blank');
            return;
        }

        try {
            setConnecting(true);

            // Request connection - onlyIfTrusted: false means prompt user
            const response = await phantom.connect({ onlyIfTrusted: false });
            const pk = response.publicKey.toBase58?.() || response.publicKey.toString();

            setPublicKey(pk);
            setConnected(true);
            console.log('Wallet connected:', pk);
        } catch (err: unknown) {
            // User rejected the connection or other error
            const errorMessage = err instanceof Error ? err.message : 'Connection failed';
            console.error('Wallet connection error:', errorMessage);

            if (errorMessage.includes('User rejected')) {
                setError('Connection rejected by user');
            } else {
                setError(errorMessage);
            }

            setConnected(false);
            setPublicKey(null);
        } finally {
            setConnecting(false);
        }
    }, []);

    const disconnect = useCallback(() => {
        const phantom = getPhantom();
        if (phantom) {
            phantom.disconnect().catch(console.error);
        }
        setConnected(false);
        setPublicKey(null);
        setError(null);
        console.log('Wallet disconnected');
    }, []);

    // Provide stable default values during SSR
    const value: WalletContextType = {
        connected: mounted ? connected : false,
        publicKey: mounted ? publicKey : null,
        phantomInstalled: mounted ? phantomInstalled : false,
        connecting: mounted ? connecting : false,
        error: mounted ? error : null,
        connect,
        disconnect,
    };

    return (
        <WalletContext.Provider value={value}>
            {children}
        </WalletContext.Provider>
    );
}
