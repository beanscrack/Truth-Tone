import { NextRequest, NextResponse } from 'next/server';

// API Configuration
const ML_API_URL = process.env.ML_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();

        if (!body.text) {
            return NextResponse.json({ error: 'No text provided' }, { status: 400 });
        }

        // Forward to ML Backend using native fetch
        const backendResponse = await fetch(`${ML_API_URL}/generate-fake`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });

        if (!backendResponse.ok) {
            const errorText = await backendResponse.text();
            console.error('Backend Error:', errorText);
            return NextResponse.json(
                { error: `Generation Failed: ${backendResponse.statusText}` },
                { status: backendResponse.status }
            );
        }

        const data = await backendResponse.json();
        return NextResponse.json(data);

    } catch (error: any) {
        console.error('Error proxying to ML backend:', error);
        return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
    }
}
