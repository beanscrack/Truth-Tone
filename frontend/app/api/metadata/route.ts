import { NextRequest, NextResponse } from 'next/server';

/**
 * NFT Metadata endpoint for TruthTone++ Verification Certificates.
 * Returns JSON metadata in the format expected by Solana/Metaplex NFT standards.
 * 
 * METAPLEX LIMITS:
 * - name: max 32 chars (we use shorter in metadata JSON, on-chain uses separate value)
 * - symbol: max 10 chars
 * 
 * Query params:
 * - h: Short audio hash (first 16 chars)
 * - s: Verification score (0-100)
 * - v: Verdict code (R=REAL, F=FAKE, etc.)
 * - t: Unix timestamp
 */
export async function GET(request: NextRequest) {
    const searchParams = request.nextUrl.searchParams;

    // Short param names to keep URI under 200 chars
    const hashShort = searchParams.get('h') || searchParams.get('audio_hash') || 'unknown';
    const score = searchParams.get('s') || searchParams.get('overall_score') || '0';
    const verdictCode = searchParams.get('v') || searchParams.get('verdict') || 'U';
    const ts = searchParams.get('t') || searchParams.get('timestamp') || Date.now().toString();

    // Expand verdict code to full text
    const verdictMap: Record<string, string> = {
        'R': 'REAL',
        'LR': 'LIKELY REAL',
        'U': 'UNCERTAIN',
        'LF': 'LIKELY FAKE',
        'F': 'FAKE',
        'REAL': 'REAL',
        'LIKELY REAL': 'LIKELY REAL',
        'UNCERTAIN': 'UNCERTAIN',
        'LIKELY FAKE': 'LIKELY FAKE',
        'FAKE': 'FAKE',
    };
    const verdict = verdictMap[verdictCode] || verdictCode;

    // Format timestamp
    const timestamp = isNaN(Number(ts)) ? ts : new Date(Number(ts)).toISOString();

    // Construct NFT metadata following Metaplex Token Metadata standard
    // Note: The on-chain name/symbol are set separately in mintVerificationNFT
    // This JSON is for off-chain metadata display
    const metadata = {
        name: "TruthTone Cert",  // Short name for display (on-chain uses "TruthTone++ Cert")
        symbol: "TTONE",         // Matches on-chain symbol
        description: "Audio authenticity certificate issued by TruthTone++ deepfake detection.",
        image: "", // Could add a generated certificate image URL
        external_url: "https://truthtone.app",
        attributes: [
            {
                trait_type: "Audio Hash",
                value: hashShort,
            },
            {
                trait_type: "Score",
                value: parseInt(score, 10),
            },
            {
                trait_type: "Verdict",
                value: verdict,
            },
            {
                trait_type: "Verified At",
                value: timestamp,
            },
            {
                trait_type: "Platform",
                value: "TruthTone++",
            },
            {
                trait_type: "Network",
                value: "devnet",
            },
        ],
        properties: {
            category: "certificate",
            creators: [],
        },
    };

    return NextResponse.json(metadata, {
        headers: {
            'Cache-Control': 'public, max-age=31536000', // Cache for 1 year
            'Content-Type': 'application/json',
        },
    });
}
