import { NextRequest, NextResponse } from 'next/server';

// API Configuration
const ML_API_URL = process.env.ML_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
    try {
        const formData = await request.formData();
        const file = formData.get('file');

        if (!file) {
            return NextResponse.json({ error: 'No file provided' }, { status: 400 });
        }

        // Forward to ML Backend using native fetch which handles FormData boundaries correctly
        const backendResponse = await fetch(`${ML_API_URL}/analyze`, {
            method: 'POST',
            body: formData,
        });

        if (!backendResponse.ok) {
            const errorText = await backendResponse.text();
            console.error('Backend Error:', errorText);
            return NextResponse.json(
                { error: `Backend Analysis Failed: ${backendResponse.statusText}` },
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
