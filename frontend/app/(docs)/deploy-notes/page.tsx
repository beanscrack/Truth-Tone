/**
 * Deployment Notes for TruthTone++
 * 
 * This page provides documentation for deploying the application to Vercel.
 * It's a server component that can be accessed at /deploy-notes
 */

export default function DeployNotesPage() {
    return (
        <main className="min-h-screen bg-[#050505] text-white font-sans p-8">
            <div className="max-w-3xl mx-auto">
                <h1 className="text-3xl font-bold mb-8">TruthTone++ Deployment Guide</h1>

                <section className="mb-8">
                    <h2 className="text-xl font-semibold mb-4 text-cyan-400">Required Environment Variables</h2>
                    <div className="bg-neutral-900 border border-white/10 rounded-lg p-6 font-mono text-sm">
                        <div className="mb-4">
                            <code className="text-green-400">ML_API_URL</code>
                            <p className="text-neutral-400 mt-1 font-sans text-sm">
                                URL of the FastAPI backend server. This is used by API routes to proxy requests.
                            </p>
                            <p className="text-neutral-500 mt-1 font-sans text-xs">
                                Example: <code>https://api.truthtone.example.com</code>
                            </p>
                            <p className="text-neutral-500 font-sans text-xs">
                                Default: <code>http://localhost:8000</code>
                            </p>
                        </div>

                        <div className="mb-4">
                            <code className="text-green-400">NEXT_PUBLIC_APP_URL</code>
                            <p className="text-neutral-400 mt-1 font-sans text-sm">
                                Public URL of the deployed Next.js app. Used for constructing NFT metadata URIs.
                            </p>
                            <p className="text-neutral-500 mt-1 font-sans text-xs">
                                Example: <code>https://truthtone.vercel.app</code>
                            </p>
                            <p className="text-neutral-500 font-sans text-xs">
                                Fallback: Uses <code>window.location.origin</code> on client
                            </p>
                        </div>
                    </div>
                </section>

                <section className="mb-8">
                    <h2 className="text-xl font-semibold mb-4 text-cyan-400">Optional Environment Variables</h2>
                    <div className="bg-neutral-900 border border-white/10 rounded-lg p-6 font-mono text-sm">
                        <div>
                            <code className="text-yellow-400">NEXT_PUBLIC_DEMO_SIGNATURE</code>
                            <p className="text-neutral-400 mt-1 font-sans text-sm">
                                Demo transaction signature for NFT certificate fallback when Phantom is not installed.
                            </p>
                            <p className="text-neutral-500 mt-1 font-sans text-xs">
                                Uses hardcoded devnet signature if not set.
                            </p>
                        </div>
                    </div>
                </section>

                <section className="mb-8">
                    <h2 className="text-xl font-semibold mb-4 text-cyan-400">Vercel Deployment Steps</h2>
                    <ol className="list-decimal list-inside space-y-3 text-neutral-300">
                        <li>Connect your GitHub repository to Vercel</li>
                        <li>Set the root directory to <code className="bg-neutral-800 px-2 py-0.5 rounded">frontend</code></li>
                        <li>Add the required environment variables in Vercel dashboard</li>
                        <li>Deploy! Vercel will automatically run <code className="bg-neutral-800 px-2 py-0.5 rounded">npm run build</code></li>
                    </ol>
                </section>

                <section className="mb-8">
                    <h2 className="text-xl font-semibold mb-4 text-cyan-400">Build Command</h2>
                    <div className="bg-neutral-900 border border-white/10 rounded-lg p-4 font-mono text-sm">
                        <code>npm run build</code>
                    </div>
                </section>

                <section className="mb-8">
                    <h2 className="text-xl font-semibold mb-4 text-cyan-400">Solana Network</h2>
                    <p className="text-neutral-300">
                        The application is configured to use <strong className="text-purple-400">Solana Devnet</strong> for
                        all blockchain operations. NFT minting and wallet connections will use devnet by default.
                    </p>
                    <p className="text-neutral-500 text-sm mt-2">
                        For mainnet deployment, update the <code className="bg-neutral-800 px-1 py-0.5 rounded">SOLANA_NETWORK</code> constant
                        in <code className="bg-neutral-800 px-1 py-0.5 rounded">utils/solana.ts</code>.
                    </p>
                </section>

                <section>
                    <h2 className="text-xl font-semibold mb-4 text-cyan-400">Backend Requirements</h2>
                    <p className="text-neutral-300">
                        The FastAPI backend must be deployed and accessible at the URL specified by <code className="bg-neutral-800 px-2 py-0.5 rounded">ML_API_URL</code>.
                        The backend must support:
                    </p>
                    <ul className="list-disc list-inside mt-3 space-y-2 text-neutral-400">
                        <li><code className="bg-neutral-800 px-1 py-0.5 rounded">POST /analyze</code> - Audio analysis endpoint</li>
                        <li><code className="bg-neutral-800 px-1 py-0.5 rounded">POST /generate-fake</code> - Fake audio generation (ElevenLabs)</li>
                    </ul>
                </section>
            </div>
        </main>
    );
}
