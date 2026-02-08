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
    SOLANA_NETWORK,
    BalanceDebugInfo
} from '@/utils/solana';
import { Award, ExternalLink, Loader2, CheckCircle, XCircle, ShieldCheck, AlertTriangle, RefreshCw } from 'lucide-react';
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

    // Real balance state
    const [balanceDebug, setBalanceDebug] = useState<BalanceDebugInfo | null>(null);
    const [isLoadingBalance, setIsLoadingBalance] = useState(false);

    const eligible = isEligibleForCertificate(analysisResult);
    const balance = balanceDebug?.solBalance ?? null;
    const balanceUnavailable = balanceDebug?.balanceUnavailable ?? false;
    const hasSufficientBalance = balance !== null && balance >= MIN_SOL_FOR_MINT;

    // Fetch balance when wallet connects
    useEffect(() => {
        if (connected && publicKey) {
            fetchBalance();
        } else {
            setBalanceDebug(null);
        }
    }, [connected, publicKey]);

    const fetchBalance = async () => {
        if (!publicKey) return;
        setIsLoadingBalance(true);
        console.log('[SolanaCertificatePanel] Fetching balance for:', publicKey);
        try {
            const debug = await getWalletBalanceWithDebug(publicKey);
            console.log('[SolanaCertificatePanel] Balance result:', debug);
            setBalanceDebug(debug);
        } catch (error) {
            console.error('[SolanaCertificatePanel] Balance fetch error:', error);
        } finally {
            setIsLoadingBalance(false);
        }
    };

    // Mint handler
    const handleMint = async () => {
        if (!publicKey || !connected) return;

        setMintStatus('confirming');
        setErrorMessage(null);

        try {
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

            const phantomProvider = windowAny.phantom?.solana || windowAny.solana;

            if (!phantomProvider) {
                throw new Error('Phantom wallet not found');
            }

            const currentPk = phantomProvider.publicKey?.toBase58?.() || phantomProvider.publicKey?.toString();
            if (!currentPk) {
                throw new Error('Wallet not connected');
            }

            console.log('[SolanaCertificatePanel] Minting with wallet:', currentPk);

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
        } catch (error) {
            console.error('[SolanaCertificatePanel] Minting failed:', error);
            setErrorMessage(error instanceof Error ? error.message : 'Minting failed. Please try again.');
            setMintStatus('error');
        }
    };

    // Eligibility checklist with real balance
    const EligibilityChecklist = () => (
        <div className="mt-3 p-3 bg-black/20 rounded-lg border border-white/5">
            <div className="flex items-center justify-between mb-2">
                <p className="text-[10px] font-semibold text-neutral-500 uppercase tracking-wider">
                    Mint Requirements
                </p>
                <span className="text-[10px] px-1.5 py-0.5 bg-purple-500/20 text-purple-400 rounded">
                    {SOLANA_NETWORK}
                </span>
            </div>
            <div className="space-y-1.5 text-xs">
                {/* Wallet connected */}
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
                {/* Solana Mainnet */}
                <div className="flex items-center gap-2">
                    <CheckCircle className="w-3.5 h-3.5 text-green-400" />
                    <span className="text-green-400">Solana Mainnet</span>
                </div>
                {/* Balance */}
                <div className="flex items-center gap-2">
                    {isLoadingBalance ? (
                        <Loader2 className="w-3.5 h-3.5 text-neutral-500 animate-spin" />
                    ) : hasSufficientBalance ? (
                        <CheckCircle className="w-3.5 h-3.5 text-green-400" />
                    ) : balanceUnavailable ? (
                        <AlertTriangle className="w-3.5 h-3.5 text-yellow-400" />
                    ) : balance !== null ? (
                        <XCircle className="w-3.5 h-3.5 text-red-400" />
                    ) : (
                        <XCircle className="w-3.5 h-3.5 text-neutral-500" />
                    )}
                    <span className={
                        isLoadingBalance ? 'text-neutral-500' :
                            hasSufficientBalance ? 'text-green-400' :
                                balanceUnavailable ? 'text-yellow-400' :
                                    balance !== null ? 'text-red-400' : 'text-neutral-500'
                    }>
                        {isLoadingBalance ? 'Loading balance...' :
                            balanceUnavailable ? 'Balance unavailable' :
                                balance !== null ? `Balance: ${balance.toFixed(4)} SOL` :
                                    'Connect wallet'}
                        {balance !== null && !hasSufficientBalance && !balanceUnavailable && (
                            <span className="text-neutral-500"> (need {MIN_SOL_FOR_MINT})</span>
                        )}
                    </span>
                    {connected && !isLoadingBalance && (
                        <button
                            onClick={fetchBalance}
                            className="ml-auto p-1 hover:bg-white/10 rounded transition-colors"
                            title="Refresh balance"
                        >
                            <RefreshCw className="w-3 h-3 text-neutral-400" />
                        </button>
                    )}
                </div>
            </div>

            {/* RPC Debug Info (dev only) */}
            {process.env.NODE_ENV === 'development' && balanceDebug && (
                <div className="mt-2 p-2 bg-black/40 border border-white/10 rounded text-[10px] font-mono text-neutral-400">
                    <div>RPC: {balanceDebug.rpcEndpoint}</div>
                    <div>Network: {balanceDebug.network}</div>
                    {balanceDebug.error && <div className="text-red-400">Error: {balanceDebug.error}</div>}
                </div>
            )}

            {/* Warning for insufficient balance */}
            {!hasSufficientBalance && balance !== null && !balanceUnavailable && (
                <div className="mt-2 pt-2 border-t border-white/5">
                    <p className="text-xs text-red-400">
                        Insufficient SOL ({balance.toFixed(4)} SOL). Minimum required: {MIN_SOL_FOR_MINT} SOL.
                    </p>
                </div>
            )}

            {/* Warning for RPC issues */}
            {balanceUnavailable && (
                <div className="mt-2 pt-2 border-t border-white/5">
                    <p className="text-xs text-yellow-400">
                        ⚠️ Could not fetch balance. You can still try minting if you have sufficient SOL.
                    </p>
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
                        <p className="text-xs text-green-400/70 mt-1">Your verification certificate NFT has been minted on Solana.</p>
                        <div className="mt-3 p-2 bg-black/20 rounded text-xs font-mono text-green-300/80 break-all">
                            {txSignature}
                        </div>
                        <a
                            href={getExplorerLink(txSignature)}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1.5 mt-3 text-xs font-medium text-green-400 hover:text-green-300 transition-colors"
                        >
                            View on Solscan
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
                        <button
                            onClick={() => setMintStatus('idle')}
                            className="mt-3 text-xs font-medium text-red-400 hover:text-red-300 underline transition-colors"
                        >
                            Try again
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    // Confirming state
    if (mintStatus === 'confirming') {
        return (
            <div className="p-4 bg-purple-500/10 border border-purple-500/20 rounded-lg">
                <div className="flex items-center gap-3">
                    <Loader2 className="w-5 h-5 text-purple-400 animate-spin" />
                    <div>
                        <p className="text-sm font-medium text-purple-400">Confirming Transaction</p>
                        <p className="text-xs text-purple-400/70 mt-0.5">
                            Please approve the transaction in your Phantom wallet...
                        </p>
                    </div>
                </div>
            </div>
        );
    }

    // Connected - show mint button
    if (connected && publicKey) {
        const canMint = (hasSufficientBalance || balanceUnavailable) && !isLoadingBalance;

        return (
            <div className="p-4 bg-gradient-to-br from-purple-500/10 to-cyan-500/10 border border-purple-500/20 rounded-lg">
                <div className="flex items-start gap-3">
                    <ShieldCheck className="w-5 h-5 text-purple-400 mt-0.5 shrink-0" />
                    <div className="flex-1">
                        <p className="text-sm font-medium text-white">Verification Certificate Available</p>
                        <p className="text-xs text-neutral-400 mt-1">
                            Mint an NFT certificate proving this audio was verified as authentic. Stored permanently on Solana.
                        </p>

                        <EligibilityChecklist />

                        <button
                            onClick={handleMint}
                            disabled={!canMint}
                            className={`mt-3 flex items-center gap-2 px-4 py-2 text-white text-sm font-medium rounded-lg transition-colors ${canMint
                                    ? 'bg-purple-600 hover:bg-purple-700'
                                    : 'bg-neutral-600 cursor-not-allowed opacity-50'
                                }`}
                        >
                            <Award className="w-4 h-4" />
                            Mint Verification Certificate
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    // Not connected - show wallet button
    return (
        <div className="p-4 bg-gradient-to-br from-purple-500/10 to-cyan-500/10 border border-purple-500/20 rounded-lg">
            <div className="flex items-start gap-3">
                <ShieldCheck className="w-5 h-5 text-purple-400 mt-0.5 shrink-0" />
                <div className="flex-1">
                    <p className="text-sm font-medium text-white">Verification Certificate Available</p>
                    <p className="text-xs text-neutral-400 mt-1">
                        Connect your Phantom wallet to mint an NFT certificate for this verified audio.
                    </p>

                    <div className="mt-3">
                        <WalletButton />
                    </div>

                    {/* Demo fallback when no Phantom */}
                    {!phantomInstalled && (
                        <div className="mt-4 pt-4 border-t border-white/10">
                            <p className="text-xs font-medium text-neutral-500 uppercase tracking-wider mb-2">Demo Certificate</p>
                            <p className="text-xs text-neutral-500 mb-2">
                                Without Phantom, here&apos;s an example of what a minted certificate looks like:
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
                                View Demo on Solscan
                                <ExternalLink className="w-3 h-3" />
                            </a>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
