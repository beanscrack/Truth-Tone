"use client";

import { useState, useEffect } from 'react';
import { useWallet } from '@/providers/WalletProvider';
import { WalletButton } from '@/components/WalletButton';
import { NormalizedAnalysisResult, isEligibleForCertificate } from '@/types/analysis';
import {
    mintVerificationNFT,
    getExplorerLink,
    getWalletBalanceWithDebug,
    MIN_SOL_FOR_MINT,
    DEMO_TX_SIGNATURE,
    RPC_ENDPOINT,
    SOLANA_NETWORK,
    BalanceDebugInfo
} from '@/utils/solana';
import { Award, ExternalLink, Loader2, CheckCircle, XCircle, ShieldCheck, AlertTriangle } from 'lucide-react';
import { PublicKey } from '@solana/web3.js';

interface SolanaCertificatePanelProps {
    analysisResult: NormalizedAnalysisResult;
}

type MintStatus = 'idle' | 'checking' | 'confirming' | 'success' | 'error';

export function SolanaCertificatePanel({ analysisResult }: SolanaCertificatePanelProps) {
    const { connected, publicKey, phantomInstalled } = useWallet();
    const [mintStatus, setMintStatus] = useState<MintStatus>('idle');
    const [txSignature, setTxSignature] = useState<string | null>(null);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);

    // Balance state with debug info
    const [balanceDebug, setBalanceDebug] = useState<BalanceDebugInfo | null>(null);
    const [isLoadingBalance, setIsLoadingBalance] = useState(false);

    const eligible = isEligibleForCertificate(analysisResult);
    const balance = balanceDebug?.solBalance ?? null;
    const hasSufficientBalance = balance !== null && balance >= MIN_SOL_FOR_MINT;

    // Fetch balance when wallet connects - with debug info
    useEffect(() => {
        if (connected && publicKey) {
            setIsLoadingBalance(true);
            console.log('[SolanaCertificatePanel] Fetching balance for:', publicKey);

            getWalletBalanceWithDebug(publicKey)
                .then((debug) => {
                    console.log('[SolanaCertificatePanel] Balance debug:', debug);
                    setBalanceDebug(debug);
                })
                .finally(() => setIsLoadingBalance(false));
        } else {
            setBalanceDebug(null);
        }
    }, [connected, publicKey]);

    // Refresh balance
    const refreshBalance = async () => {
        if (publicKey) {
            setIsLoadingBalance(true);
            const debug = await getWalletBalanceWithDebug(publicKey);
            setBalanceDebug(debug);
            setIsLoadingBalance(false);
        }
    };

    const handleMint = async () => {
        if (!publicKey || !connected) return;

        setMintStatus('checking');
        setErrorMessage(null);

        try {
            // Refresh balance first
            console.log('[SolanaCertificatePanel] Pre-mint balance check...');
            const debug = await getWalletBalanceWithDebug(publicKey);
            setBalanceDebug(debug);

            if (debug.error) {
                setErrorMessage(`Balance check failed: ${debug.error}`);
                setMintStatus('error');
                return;
            }

            if (debug.solBalance < MIN_SOL_FOR_MINT) {
                setErrorMessage(
                    `Insufficient devnet SOL (${debug.solBalance.toFixed(4)} SOL). ` +
                    `You need at least ${MIN_SOL_FOR_MINT} SOL. ` +
                    `Get free devnet SOL at faucet.solana.com`
                );
                setMintStatus('error');
                return;
            }

            setMintStatus('confirming');

            // Get Phantom provider for signing
            const windowAny = window as unknown as {
                solana?: {
                    publicKey: { toString(): string; toBase58(): string };
                    signTransaction: (tx: unknown) => Promise<unknown>;
                    signAllTransactions: (txs: unknown[]) => Promise<unknown[]>;
                };
                phantom?: {
                    solana?: {
                        publicKey: { toString(): string; toBase58(): string };
                        signTransaction: (tx: unknown) => Promise<unknown>;
                        signAllTransactions: (txs: unknown[]) => Promise<unknown[]>;
                    }
                }
            };

            // Try both injection methods
            const phantomProvider = windowAny.phantom?.solana || windowAny.solana;

            if (!phantomProvider) {
                throw new Error('Phantom wallet not found');
            }

            // Use the CURRENT public key from provider, not cached
            const currentPk = phantomProvider.publicKey?.toBase58?.() || phantomProvider.publicKey?.toString();
            if (!currentPk) {
                throw new Error('Wallet not connected');
            }

            console.log('[SolanaCertificatePanel] Using wallet address:', currentPk);

            const wallet = {
                publicKey: new PublicKey(currentPk),
                signTransaction: phantomProvider.signTransaction.bind(phantomProvider),
                signAllTransactions: phantomProvider.signAllTransactions?.bind(phantomProvider),
            };

            const result = await mintVerificationNFT(wallet, {
                audioHash: analysisResult.audio_hash || `sha256:${Date.now()}`,
                overallScore: analysisResult.overall_score,
                verdict: analysisResult.verdict,
                timestamp: new Date().toISOString(),
            });

            setTxSignature(result.signature);
            setMintStatus('success');

            // Refresh balance after successful mint
            await refreshBalance();
        } catch (error) {
            console.error('[SolanaCertificatePanel] Minting failed:', error);
            setErrorMessage(error instanceof Error ? error.message : 'Minting failed. Please try again.');
            setMintStatus('error');
        }
    };

    // DEV-only debug info component
    const DebugInfo = () => {
        if (process.env.NODE_ENV !== 'development') return null;
        if (!balanceDebug) return null;

        return (
            <div className="mt-2 p-2 bg-black/40 border border-yellow-500/30 rounded text-[10px] font-mono text-yellow-400/80">
                <div className="font-bold text-yellow-400 mb-1">🔧 DEV DEBUG</div>
                <div>Wallet: <span className="text-neutral-300">{balanceDebug.publicKey.slice(0, 8)}...{balanceDebug.publicKey.slice(-8)}</span></div>
                <div>RPC: <span className="text-neutral-300">{balanceDebug.rpcEndpoint}</span></div>
                <div>Network: <span className="text-cyan-400">{balanceDebug.network}</span></div>
                <div>Lamports: <span className="text-neutral-300">{balanceDebug.lamports.toLocaleString()}</span></div>
                <div>SOL: <span className="text-green-400">{balanceDebug.solBalance.toFixed(6)}</span></div>
                <div>Fetched: <span className="text-neutral-500">{balanceDebug.timestamp}</span></div>
                {balanceDebug.error && (
                    <div className="text-red-400">Error: {balanceDebug.error}</div>
                )}
                <button
                    onClick={refreshBalance}
                    className="mt-1 px-2 py-0.5 bg-yellow-600/30 hover:bg-yellow-600/50 rounded text-yellow-300"
                >
                    Refresh Balance
                </button>
            </div>
        );
    };

    // Eligibility checklist component
    const EligibilityChecklist = () => (
        <div className="mt-3 p-3 bg-black/20 rounded-lg border border-white/5">
            <p className="text-[10px] font-semibold text-neutral-500 uppercase tracking-wider mb-2">
                Mint Requirements
            </p>
            <div className="space-y-1.5 text-xs">
                <div className="flex items-center gap-2">
                    {connected ? (
                        <CheckCircle className="w-3.5 h-3.5 text-green-400" />
                    ) : (
                        <XCircle className="w-3.5 h-3.5 text-neutral-500" />
                    )}
                    <span className={connected ? 'text-green-400' : 'text-neutral-500'}>
                        Wallet connected
                    </span>
                </div>
                <div className="flex items-center gap-2">
                    <CheckCircle className="w-3.5 h-3.5 text-green-400" />
                    <span className="text-green-400">
                        Devnet network
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 bg-purple-500/20 text-purple-400 rounded">
                        {SOLANA_NETWORK}
                    </span>
                </div>
                <div className="flex items-center gap-2">
                    {isLoadingBalance ? (
                        <Loader2 className="w-3.5 h-3.5 text-neutral-500 animate-spin" />
                    ) : hasSufficientBalance ? (
                        <CheckCircle className="w-3.5 h-3.5 text-green-400" />
                    ) : balance !== null ? (
                        <XCircle className="w-3.5 h-3.5 text-red-400" />
                    ) : (
                        <XCircle className="w-3.5 h-3.5 text-neutral-500" />
                    )}
                    <span className={
                        isLoadingBalance ? 'text-neutral-500' :
                            hasSufficientBalance ? 'text-green-400' :
                                balance !== null ? 'text-red-400' : 'text-neutral-500'
                    }>
                        Balance: {
                            isLoadingBalance ? 'Loading...' :
                                balance !== null ? `${balance.toFixed(4)} SOL` :
                                    'Connect wallet'
                        }
                        {balance !== null && !hasSufficientBalance && (
                            <span className="text-neutral-500"> (need {MIN_SOL_FOR_MINT})</span>
                        )}
                    </span>
                </div>
            </div>

            <DebugInfo />

            {connected && balance !== null && !hasSufficientBalance && (
                <div className="mt-2 pt-2 border-t border-white/5">
                    <a
                        href="https://faucet.solana.com/"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1.5 text-xs font-medium text-cyan-400 hover:text-cyan-300"
                    >
                        Get free devnet SOL →
                    </a>
                </div>
            )}
        </div>
    );

    // Not eligible
    if (!eligible) {
        return (
            <div className="p-4 bg-neutral-900/50 border border-white/[0.06] rounded-lg">
                <div className="flex items-center gap-3 text-neutral-500">
                    <Award className="w-5 h-5" />
                    <div>
                        <p className="text-sm font-medium">Certificate Not Available</p>
                        <p className="text-xs mt-0.5">
                            NFT certificates are only available for audio verified as REAL or LIKELY REAL with a score of 85% or higher.
                        </p>
                    </div>
                </div>
            </div>
        );
    }

    // Success state
    if (mintStatus === 'success' && txSignature) {
        return (
            <div className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                <div className="flex items-start gap-3">
                    <CheckCircle className="w-5 h-5 text-green-400 mt-0.5 shrink-0" />
                    <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-green-400">Certificate Minted!</p>
                        <p className="text-xs text-green-400/70 mt-1">Your verification certificate NFT has been minted on Solana devnet.</p>
                        <div className="mt-3 p-2 bg-black/20 rounded text-xs font-mono text-green-300/80 break-all">
                            {txSignature}
                        </div>
                        <a
                            href={getExplorerLink(txSignature)}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1.5 mt-3 text-xs font-medium text-green-400 hover:text-green-300 transition-colors"
                        >
                            View on Solana Explorer
                            <ExternalLink className="w-3 h-3" />
                        </a>
                    </div>
                </div>
            </div>
        );
    }

    // Error state
    if (mintStatus === 'error') {
        return (
            <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
                <div className="flex items-start gap-3">
                    <XCircle className="w-5 h-5 text-red-400 mt-0.5 shrink-0" />
                    <div className="flex-1">
                        <p className="text-sm font-medium text-red-400">Minting Failed</p>
                        <p className="text-xs text-red-400/70 mt-1">{errorMessage}</p>

                        <DebugInfo />

                        {/* Show faucet link if balance issue */}
                        {errorMessage?.toLowerCase().includes('insufficient') && (
                            <a
                                href="https://faucet.solana.com/"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1.5 mt-2 text-xs font-medium text-cyan-400 hover:text-cyan-300"
                            >
                                Get free devnet SOL at faucet.solana.com →
                            </a>
                        )}

                        <button
                            onClick={() => {
                                setMintStatus('idle');
                                refreshBalance();
                            }}
                            className="mt-3 text-xs font-medium text-red-400 hover:text-red-300 underline transition-colors"
                        >
                            Try again
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    // Checking/Confirming state
    if (mintStatus === 'checking' || mintStatus === 'confirming') {
        return (
            <div className="p-4 bg-purple-500/10 border border-purple-500/20 rounded-lg">
                <div className="flex items-center gap-3">
                    <Loader2 className="w-5 h-5 text-purple-400 animate-spin" />
                    <div>
                        <p className="text-sm font-medium text-purple-400">
                            {mintStatus === 'checking' ? 'Checking Balance...' : 'Confirming Transaction'}
                        </p>
                        <p className="text-xs text-purple-400/70 mt-0.5">
                            {mintStatus === 'checking'
                                ? 'Verifying wallet balance before minting...'
                                : 'Please approve the transaction in your Phantom wallet...'}
                        </p>
                    </div>
                </div>
            </div>
        );
    }

    // Connected - show mint button with eligibility checklist
    if (connected && publicKey) {
        const canMint = hasSufficientBalance && !isLoadingBalance;

        return (
            <div className="p-4 bg-gradient-to-br from-purple-500/10 to-cyan-500/10 border border-purple-500/20 rounded-lg">
                <div className="flex items-start gap-3">
                    <ShieldCheck className="w-5 h-5 text-purple-400 mt-0.5 shrink-0" />
                    <div className="flex-1">
                        <p className="text-sm font-medium text-white">Verification Certificate Available</p>
                        <p className="text-xs text-neutral-400 mt-1">
                            Mint an NFT certificate proving this audio was verified as authentic. Stored permanently on Solana devnet.
                        </p>

                        <EligibilityChecklist />

                        {!hasSufficientBalance && !isLoadingBalance && (
                            <div className="mt-3 p-2 bg-yellow-500/10 border border-yellow-500/20 rounded flex items-start gap-2">
                                <AlertTriangle className="w-4 h-4 text-yellow-400 shrink-0 mt-0.5" />
                                <p className="text-xs text-yellow-400">
                                    Insufficient devnet SOL. Fund your wallet using a Solana devnet faucet before minting.
                                </p>
                            </div>
                        )}

                        <button
                            onClick={handleMint}
                            disabled={!canMint}
                            className="mt-3 flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-purple-600"
                        >
                            <Award className="w-4 h-4" />
                            {isLoadingBalance ? 'Checking Balance...' : 'Mint Verification Certificate'}
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    // Not connected - show wallet button + demo fallback
    return (
        <div className="p-4 bg-gradient-to-br from-purple-500/10 to-cyan-500/10 border border-purple-500/20 rounded-lg">
            <div className="flex items-start gap-3">
                <ShieldCheck className="w-5 h-5 text-purple-400 mt-0.5 shrink-0" />
                <div className="flex-1">
                    <p className="text-sm font-medium text-white">Verification Certificate Available</p>
                    <p className="text-xs text-neutral-400 mt-1">
                        Connect your Phantom wallet to mint an NFT certificate for this verified audio.
                    </p>

                    <EligibilityChecklist />

                    <div className="mt-3">
                        <WalletButton />
                    </div>

                    {/* Demo mode fallback */}
                    {!phantomInstalled && (
                        <div className="mt-4 pt-4 border-t border-white/10">
                            <p className="text-xs font-medium text-neutral-500 uppercase tracking-wider mb-2">Demo Mode Certificate</p>
                            <p className="text-xs text-neutral-500 mb-2">
                                Without Phantom, here&apos;s an example of what a minted certificate looks like on devnet:
                            </p>
                            <div className="p-2 bg-black/20 rounded text-xs font-mono text-neutral-400 break-all">
                                {DEMO_TX_SIGNATURE}
                            </div>
                            <a
                                href={getExplorerLink(DEMO_TX_SIGNATURE)}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1.5 mt-2 text-xs font-medium text-purple-400 hover:text-purple-300 transition-colors"
                            >
                                View Demo Certificate on Explorer
                                <ExternalLink className="w-3 h-3" />
                            </a>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
