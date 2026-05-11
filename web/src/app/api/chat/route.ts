import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.RAG_BACKEND_URL || 'http://localhost:8000';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { message, history } = body;

    if (!message?.trim()) {
      return NextResponse.json({ error: 'Message is required' }, { status: 400 });
    }

    const response = await fetch(`${BACKEND_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history: history ?? [] }),
      // Give the RAG pipeline up to 120s to respond (scraping can be slow)
      signal: AbortSignal.timeout(120_000),

    });

    // Stream the response body directly from the RAG backend to the frontend
    return new Response(response.body, {
      headers: {
        'Content-Type': 'application/x-ndjson',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });

  } catch (err: any) {
    console.error('[/api/chat error]', err);
    return NextResponse.json(
      { error: err?.message || 'Unexpected error' },
      { status: 500 },
    );
  }
}
