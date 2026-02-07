import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';

// API Configuration
const ML_API_URL = process.env.ML_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
    try {
        const formData = await request.formData();
        const file = formData.get('file');

        if (!file) {
            return NextResponse.json({ error: 'No file provided' }, { status: 400 });
        }

        // Forward to ML Backend
        // Note: We need to reconstruct FormData to send via axios
        const backendFormData = new FormData();
        backendFormData.append('file', file);

        const response = await axios.post(`${ML_API_URL}/analyze`, backendFormData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });

        return NextResponse.json(response.data);
    } catch (error: any) {
        console.error('Error proxying to ML backend:', error.message);
        if (error.response) {
            return NextResponse.json(error.response.data, { status: error.response.status });
        }
        return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
    }
}
