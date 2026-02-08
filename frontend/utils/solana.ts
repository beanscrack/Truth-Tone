import { Connection, PublicKey, clusterApiUrl, LAMPORTS_PER_SOL } from '@solana/web3.js';
import { Metaplex, walletAdapterIdentity } from '@metaplex-foundation/js';

/**
 * ================================================================================
 * SOLANA CONFIGURATION
 * ================================================================================
 * 
 * NETWORK: devnet (for hackathon demo)
 * RPC: https://api.devnet.solana.com
 * 
 * METAPLEX TOKEN METADATA LIMITS:
 * - name: max 32 characters
 * - symbol: max 10 characters  
 * - uri: max 200 characters
 * - sellerFeeBasisPoints: 0-10000
 * 
 * IMPORTANT NOTES:
 * - Phantom wallet must be on devnet to sign transactions
 * - Users need devnet SOL from a faucet (https://faucet.solana.com/)
 * - Minimum balance: ~0.015 SOL for NFT mint transaction
 * - This is entirely client-side; backend does NOT need wallet access
 * ================================================================================
 */

// Configuration - EXPLICITLY using devnet
export const SOLANA_NETWORK = 'devnet' as const;
export const RPC_ENDPOINT = clusterApiUrl(SOLANA_NETWORK);

// Minimum SOL required for minting (includes rent + tx fees)
export const MIN_SOL_FOR_MINT = 0.02;

// Metaplex Token Metadata limits
const METAPLEX_LIMITS = {
    NAME_MAX: 32,
    SYMBOL_MAX: 10,
    URI_MAX: 200,
} as const;

// NFT metadata constants (within Metaplex limits)
const NFT_NAME = "TruthTone++ Cert";  // 16 chars (< 32)
const NFT_SYMBOL = "TTONE";            // 5 chars (< 10)

// Create a single shared connection instance for ALL Solana operations
export const connection = new Connection(RPC_ENDPOINT, {
    commitment: 'confirmed',
    confirmTransactionInitialTimeout: 60000,
});

// Devnet explorer URL
const SOLANA_EXPLORER_BASE = 'https://explorer.solana.com';

/**
 * Get Solana explorer link for a transaction signature
 */
export function getExplorerLink(signature: string): string {
    return `${SOLANA_EXPLORER_BASE}/tx/${signature}?cluster=devnet`;
}

/**
 * Get Solana explorer link for an account/address
 */
export function getAddressExplorerLink(address: string): string {
    return `${SOLANA_EXPLORER_BASE}/address/${address}?cluster=devnet`;
}

/**
 * Debug info for troubleshooting RPC issues
 */
export interface BalanceDebugInfo {
    publicKey: string;
    rpcEndpoint: string;
    network: string;
    lamports: number;
    solBalance: number;
    timestamp: string;
    error?: string;
}

/**
 * Get the SOL balance for a wallet address with debug info
 */
export async function getWalletBalanceWithDebug(publicKey: string): Promise<BalanceDebugInfo> {
    const debugInfo: BalanceDebugInfo = {
        publicKey,
        rpcEndpoint: RPC_ENDPOINT,
        network: SOLANA_NETWORK,
        lamports: 0,
        solBalance: 0,
        timestamp: new Date().toISOString(),
    };

    try {
        const pk = new PublicKey(publicKey);
        console.log(`[Solana] Fetching balance for ${pk.toBase58()}`);
        console.log(`[Solana] Using RPC: ${RPC_ENDPOINT}`);

        const lamports = await connection.getBalance(pk);
        const solBalance = lamports / LAMPORTS_PER_SOL;

        debugInfo.lamports = lamports;
        debugInfo.solBalance = solBalance;

        console.log(`[Solana] Balance: ${lamports} lamports = ${solBalance} SOL`);
        return debugInfo;
    } catch (error) {
        const errorMsg = error instanceof Error ? error.message : String(error);
        console.error('[Solana] Balance fetch failed:', errorMsg);
        debugInfo.error = errorMsg;
        return debugInfo;
    }
}

/**
 * Get the SOL balance for a wallet address (simple version)
 */
export async function getWalletBalance(publicKey: string): Promise<number> {
    const debug = await getWalletBalanceWithDebug(publicKey);
    return debug.solBalance;
}

/**
 * Check if wallet has enough SOL for minting
 */
export async function checkMintEligibility(publicKey: string): Promise<{
    balance: number;
    hasSufficientBalance: boolean;
    minRequired: number;
    debug: BalanceDebugInfo;
}> {
    const debug = await getWalletBalanceWithDebug(publicKey);
    return {
        balance: debug.solBalance,
        hasSufficientBalance: debug.solBalance >= MIN_SOL_FOR_MINT,
        minRequired: MIN_SOL_FOR_MINT,
        debug,
    };
}

/**
 * Initialize Metaplex with the connected wallet adapter
 */
export const initMetaplex = (wallet: any) => {
    return Metaplex.make(connection).use(walletAdapterIdentity(wallet));
};

/**
 * Demo mode constants for when Phantom isn't installed
 */
export const DEMO_TX_SIGNATURE = '5KtPPmAhYe1Y1gZ3Q5JGnUxH5GbK8qR6JdXV3cGvpMWj9K8EFzmzT7HkX4dVb2JnJz6RPLmwPxGy4F9nLxY1cDnZ';

interface MintMetadata {
    audioHash: string;
    overallScore: number;
    verdict: string;
    timestamp: string;
}

/**
 * Map verdict to short code for compact URI
 */
function getVerdictCode(verdict: string): string {
    const map: Record<string, string> = {
        'REAL': 'R',
        'LIKELY REAL': 'LR',
        'UNCERTAIN': 'U',
        'LIKELY FAKE': 'LF',
        'FAKE': 'F',
    };
    return map[verdict.toUpperCase()] || 'U';
}

/**
 * Build metadata URI for the NFT
 * Uses short query params to stay under 200 char limit
 */
function buildMetadataUri(metadata: MintMetadata): string {
    const baseUrl = typeof window !== 'undefined'
        ? (process.env.NEXT_PUBLIC_APP_URL || window.location.origin)
        : 'http://localhost:3000';

    // Shorten audio hash to first 16 chars
    const hashShort = metadata.audioHash.replace('sha256:', '').substring(0, 16);

    // Use Unix timestamp instead of ISO string
    const ts = Date.parse(metadata.timestamp) || Date.now();

    // Use short param names: h=hash, s=score, v=verdict, t=timestamp
    const params = new URLSearchParams({
        h: hashShort,
        s: metadata.overallScore.toString(),
        v: getVerdictCode(metadata.verdict),
        t: ts.toString(),
    });

    const uri = `${baseUrl}/api/metadata?${params.toString()}`;

    // Log URI for debugging
    console.log(`[Solana] Metadata URI (${uri.length} chars): ${uri}`);

    return uri;
}

/**
 * Validate NFT metadata fields before minting
 * Returns null if valid, or error message if invalid
 */
export function validateNftMetadata(name: string, symbol: string, uri: string): string | null {
    if (name.length > METAPLEX_LIMITS.NAME_MAX) {
        return `NFT name too long: ${name.length} chars (max ${METAPLEX_LIMITS.NAME_MAX})`;
    }
    if (symbol.length > METAPLEX_LIMITS.SYMBOL_MAX) {
        return `NFT symbol too long: ${symbol.length} chars (max ${METAPLEX_LIMITS.SYMBOL_MAX})`;
    }
    if (uri.length > METAPLEX_LIMITS.URI_MAX) {
        return `Metadata URI too long: ${uri.length} chars (max ${METAPLEX_LIMITS.URI_MAX})`;
    }
    return null;
}

/**
 * Parse Solana transaction errors into user-friendly messages
 */
function parseTransactionError(error: unknown): string {
    const errorMessage = error instanceof Error ? error.message : String(error);

    if (errorMessage.includes('insufficient lamports') ||
        errorMessage.includes('Insufficient funds')) {
        return 'Insufficient devnet SOL. Please fund your wallet using https://faucet.solana.com/';
    }

    if (errorMessage.includes('Transaction simulation failed') ||
        errorMessage.includes('Simulation failed')) {
        return 'Transaction simulation failed. Ensure your wallet has devnet SOL and is connected to devnet.';
    }

    if (errorMessage.includes('User rejected')) {
        return 'Transaction was rejected by user.';
    }

    if (errorMessage.includes('Blockhash not found') ||
        errorMessage.includes('block height exceeded')) {
        return 'Transaction expired. Please try again.';
    }

    if (errorMessage.includes('Network request failed') ||
        errorMessage.includes('fetch failed')) {
        return 'Network error. Please check your connection and try again.';
    }

    if (errorMessage.includes('Name too long') ||
        errorMessage.includes('0xb')) {
        return 'NFT metadata field too long. This is a bug - please report it.';
    }

    return errorMessage;
}

/**
 * Mint a verification NFT for the audio file
 * @param wallet The connected wallet adapter (must have signTransaction)
 * @param metadata Metadata object containing audio analysis results
 * @returns Transaction signature and explorer link
 */
export const mintVerificationNFT = async (wallet: {
    publicKey: PublicKey;
    signTransaction: (tx: any) => Promise<any>;
    signAllTransactions?: (txs: any[]) => Promise<any[]>;
}, metadata: MintMetadata): Promise<{ signature: string; explorerLink: string }> => {
    try {
        console.log('[Solana] Starting NFT mint...');
        console.log('[Solana] Wallet:', wallet.publicKey.toBase58());
        console.log('[Solana] RPC:', RPC_ENDPOINT);

        // Build metadata URI
        const metadataUri = buildMetadataUri(metadata);

        // Validate metadata fields BEFORE attempting to mint
        const validationError = validateNftMetadata(NFT_NAME, NFT_SYMBOL, metadataUri);
        if (validationError) {
            console.error('[Solana] Metadata validation failed:', validationError);
            throw new Error(validationError);
        }

        console.log('[Solana] Metadata validated:');
        console.log(`  Name: "${NFT_NAME}" (${NFT_NAME.length}/${METAPLEX_LIMITS.NAME_MAX} chars)`);
        console.log(`  Symbol: "${NFT_SYMBOL}" (${NFT_SYMBOL.length}/${METAPLEX_LIMITS.SYMBOL_MAX} chars)`);
        console.log(`  URI: ${metadataUri.length}/${METAPLEX_LIMITS.URI_MAX} chars`);

        // Pre-flight balance check
        const balanceCheck = await checkMintEligibility(wallet.publicKey.toBase58());
        console.log('[Solana] Balance check:', balanceCheck);

        if (!balanceCheck.hasSufficientBalance) {
            throw new Error(
                `Insufficient devnet SOL (${balanceCheck.balance.toFixed(4)} SOL). ` +
                `Minimum required: ${MIN_SOL_FOR_MINT} SOL. ` +
                `Fund your wallet at https://faucet.solana.com/`
            );
        }

        const metaplex = initMetaplex(wallet);

        // Mint the NFT with validated, short name and symbol
        const { nft, response } = await metaplex.nfts().create({
            uri: metadataUri,
            name: NFT_NAME,           // "TruthTone++ Cert" (16 chars)
            symbol: NFT_SYMBOL,       // "TTONE" (5 chars)
            sellerFeeBasisPoints: 0,
            creators: [
                { address: wallet.publicKey, share: 100 }
            ],
            isMutable: false,
        });

        const signature = response.signature;
        console.log('[Solana] NFT minted! Signature:', signature);

        return {
            signature,
            explorerLink: getExplorerLink(signature),
        };
    } catch (error) {
        console.error("[Solana] Minting failed:", error);
        throw new Error(parseTransactionError(error));
    }
};
