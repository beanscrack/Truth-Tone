import { Connection, PublicKey, clusterApiUrl } from '@solana/web3.js';
import { Metaplex, walletAdapterIdentity } from '@metaplex-foundation/js';

// Configuration
const SOLANA_NETWORK = 'devnet'; // Use 'devnet' for hackathon
const RPC_ENDPOINT = clusterApiUrl(SOLANA_NETWORK);

export const connection = new Connection(RPC_ENDPOINT);

/**
 * Initialize Metaplex with the connected wallet adapter
 */
export const initMetaplex = (wallet: any) => {
    return Metaplex.make(connection).use(walletAdapterIdentity(wallet));
};

/**
 * Mint a verification NFT for the audio file
 * @param wallet The connected wallet adapter
 * @param metadata Metadata object containing audio analysis results
 */
export const mintVerificationNFT = async (wallet: any, metadata: {
    filename: string,
    hash: string,
    confidence: number,
    verdict: string,
    timestamp: string
}) => {
    try {
        const metaplex = initMetaplex(wallet);

        // 1. Upload Metadata JSON (In production, upload to Arweave via Bundlr/Irys)
        // For hackathon speed/cost, we might simplify or use a free gateway if possible,
        // or just mock the URI if we don't have devnet SOL for storage fees.

        // For now, let's assume we construct a metadata object to mint.
        // Note: To upload off-chain metadata, we need storage. 
        // We will use a Mock URI for now to save setup time, but in real implementation use:
        // const { uri } = await metaplex.nfts().uploadMetadata({ ... });

        const fakeUri = "https://example.com/metadata.json";

        // 2. Mint the NFT
        const { nft } = await metaplex.nfts().create({
            uri: fakeUri,
            name: "TruthTone Verified",
            sellerFeeBasisPoints: 0,
            symbol: "TRUTH",
            creators: [
                { address: wallet.publicKey, share: 100 }
            ],
            isMutable: false,
        });

        return {
            signature: nft.address.toString(), // Return the mint address or tx signature
            mintAddress: nft.address.toString()
        };
    } catch (error) {
        console.error("Minting failed:", error);
        throw error;
    }
};
