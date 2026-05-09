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
      // Give the RAG pipeline up to 60s to respond
      signal: AbortSignal.timeout(60_000),
    });

    if (!response.ok) {
      const err = await response.text();
      console.error('[RAG backend error]', err);
      return NextResponse.json(
        { error: 'RAG backend returned an error', detail: err },
        { status: 502 },
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (err: any) {
    console.error('[/api/chat error]', err);
    return NextResponse.json(
      { error: err?.message || 'Unexpected error' },
      { status: 500 },
    );
  }
}
