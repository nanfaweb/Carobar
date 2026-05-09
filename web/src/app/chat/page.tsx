"use client";

import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  Car, Send, Star, StarOff, User, Bot, Loader2,
  LogOut, Home, Menu, X, MessageSquare, AlertCircle
} from 'lucide-react';
import { supabase } from '@/lib/supabase';

// ─── Types ───────────────────────────────────────────────────────────────────

type CarListing = {
  listing_id: number;
  title: string;
  make: string;
  model: string;
  year: number;
  price_pkr: number;
  price_display: string;
  mileage_km: number;
  fuel_type: string;
  transmission: string;
  location: string;
  hero_image?: string;
  listing_url?: string;
  similarity?: number;
};

type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  cars?: CarListing[];
  error?: boolean;
};

type HistoryEntry = { role: 'user' | 'assistant'; content: string };

// ─── Component ───────────────────────────────────────────────────────────────

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '0',
      role: 'assistant',
      content:
        'Hi! I\'m your Carobar AI assistant, powered by real car listings. Try asking me:\n\n• "Show me Honda Civics under PKR 5 million"\n• "Best automatic cars in Lahore"\n• "Compare Toyota Corolla vs Honda City"',
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [savedIds, setSavedIds] = useState<Set<number>>(new Set());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load already-saved favorites so the star shows correctly
  useEffect(() => {
    (async () => {
      const { data: userData } = await supabase.auth.getUser();
      if (!userData.user) return;
      const { data } = await supabase
        .from('favorites')
        .select('listing_id')
        .eq('user_id', userData.user.id);
      if (data) setSavedIds(new Set(data.map((r: any) => r.listing_id)));
    })();
  }, []);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    router.push('/login');
  };

  // Build conversation history for the backend (exclude the first greeting)
  const buildHistory = (msgs: Message[]): HistoryEntry[] =>
    msgs
      .slice(1) // skip the initial greeting
      .filter((m) => !m.error)
      .map((m) => ({ role: m.role, content: m.content }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userContent = input.trim();
    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: userContent,
    };

    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);
    setInput('');
    setIsLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userContent,
          history: buildHistory(nextMessages.slice(0, -1)),
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(errData.detail || errData.error || `HTTP ${res.status}`);
      }

      const data = await res.json();

      const botMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.reply,
        cars: data.cars ?? [],
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err: any) {
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Sorry, I ran into a problem: ${err.message}. Make sure the RAG backend is running (\`uvicorn backend.main:app --reload\`).`,
        error: true,
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const toggleFavorite = async (car: CarListing) => {
    const { data: userData } = await supabase.auth.getUser();
    if (!userData.user) {
      alert('Please sign in to save favorites.');
      router.push('/login');
      return;
    }

    const isSaved = savedIds.has(car.listing_id);

    if (isSaved) {
      const { error } = await supabase
        .from('favorites')
        .delete()
        .eq('user_id', userData.user.id)
        .eq('listing_id', car.listing_id);
      if (!error) {
        setSavedIds((prev) => {
          const next = new Set(prev);
          next.delete(car.listing_id);
          return next;
        });
      }
    } else {
      const { error } = await supabase.from('favorites').insert({
        user_id: userData.user.id,
        listing_id: car.listing_id,
      });
      if (!error) {
        setSavedIds((prev) => new Set([...prev, car.listing_id]));
      } else if (error.code === '23505') {
        // Already exists — just update local state
        setSavedIds((prev) => new Set([...prev, car.listing_id]));
      } else {
        console.error(error);
      }
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-[#050505] text-white">
      {/* Mobile Sidebar Overlay */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-40 md:hidden backdrop-blur-sm"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 w-64 glass-panel border-r border-white/10 transform transition-transform duration-300 ease-in-out
          md:relative md:translate-x-0 flex flex-col ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}
      >
        <div className="p-4 border-b border-white/10 flex items-center justify-between shrink-0">
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#8b5cf6] to-blue-500 flex items-center justify-center">
              <Car className="text-white w-5 h-5" />
            </div>
            <span className="font-bold text-xl text-white">Carobar</span>
          </Link>
          <button className="md:hidden text-gray-400 hover:text-white" onClick={() => setIsSidebarOpen(false)}>
            <X className="w-5 h-5" />
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-1">
          <Link href="/" className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-gray-400 hover:bg-white/5 hover:text-white transition-colors">
            <Home className="w-5 h-5" />
            Home
          </Link>
          <div className="flex items-center gap-3 px-3 py-2.5 rounded-xl bg-[#8b5cf6]/10 text-[#8b5cf6] font-medium border border-[#8b5cf6]/20">
            <MessageSquare className="w-5 h-5" />
            AI Chat
          </div>
          <Link href="/favorites" className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-gray-400 hover:bg-white/5 hover:text-white transition-colors">
            <Star className="w-5 h-5" />
            Saved Cars
          </Link>
        </nav>

        <div className="p-4 border-t border-white/10 shrink-0">
          <button
            onClick={handleSignOut}
            className="flex items-center gap-3 px-3 py-2.5 w-full rounded-xl text-gray-400 hover:bg-white/5 hover:text-red-400 transition-colors"
          >
            <LogOut className="w-5 h-5" />
            Sign Out
          </button>
        </div>
      </aside>

      {/* Chat Area */}
      <main className="flex-1 flex flex-col min-w-0 h-full">
        {/* Header */}
        <header className="h-16 glass-panel border-b border-white/10 flex items-center px-4 shrink-0 z-10 justify-between">
          <div className="flex items-center gap-3 md:hidden">
            <button onClick={() => setIsSidebarOpen(true)} className="p-2 text-gray-400 hover:text-white rounded-xl hover:bg-white/5">
              <Menu className="w-6 h-6" />
            </button>
            <span className="font-bold text-white">Carobar</span>
          </div>
          <div className="hidden md:block" />
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <span className="hidden sm:inline">RAG AI Active</span>
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
          {messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              msg={msg}
              savedIds={savedIds}
              onToggleFavorite={toggleFavorite}
            />
          ))}

          {isLoading && (
            <div className="flex gap-4 max-w-4xl mx-auto">
              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#8b5cf6] to-blue-500 flex items-center justify-center shrink-0 shadow-lg shadow-[#8b5cf6]/20">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div className="px-5 py-4 rounded-2xl rounded-tl-sm glass-panel flex items-center gap-2">
                <span className="w-2 h-2 bg-[#8b5cf6] rounded-full animate-bounce" />
                <span className="w-2 h-2 bg-[#8b5cf6] rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                <span className="w-2 h-2 bg-[#8b5cf6] rounded-full animate-bounce" style={{ animationDelay: '0.4s' }} />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="shrink-0 p-4 bg-gradient-to-t from-[#050505] via-[#050505]/90 to-transparent pt-8 z-20">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto flex items-center relative">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about any car — make, budget, city…"
              disabled={isLoading}
              className="w-full glass-input pl-5 pr-14 py-4 rounded-full outline-none shadow-lg text-white placeholder-gray-500"
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="absolute right-2 p-2.5 rounded-full bg-[#8b5cf6] text-white hover:bg-[#7c3aed] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
          <p className="text-center mt-2 text-xs text-gray-600">
            AI answers are grounded in live database listings — always verify prices before purchasing.
          </p>
        </div>
      </main>
    </div>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function MessageBubble({
  msg,
  savedIds,
  onToggleFavorite,
}: {
  msg: Message;
  savedIds: Set<number>;
  onToggleFavorite: (car: CarListing) => void;
}) {
  const isUser = msg.role === 'user';

  return (
    <div className={`flex gap-4 max-w-4xl mx-auto ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 ${
          isUser
            ? 'bg-white/10 text-gray-300'
            : 'bg-gradient-to-br from-[#8b5cf6] to-blue-500 text-white shadow-lg shadow-[#8b5cf6]/20'
        }`}
      >
        {isUser ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
      </div>

      {/* Bubble + cards */}
      <div className={`flex flex-col gap-3 min-w-0 max-w-[85%] ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`px-5 py-3.5 rounded-2xl whitespace-pre-wrap leading-relaxed text-sm ${
            isUser
              ? 'bg-[#7c3aed] text-white rounded-tr-sm'
              : msg.error
              ? 'bg-red-500/10 border border-red-500/20 text-red-300 rounded-tl-sm flex items-start gap-2'
              : 'glass-panel text-gray-200 rounded-tl-sm'
          }`}
        >
          {msg.error && <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />}
          {msg.content}
        </div>

        {/* Car result cards */}
        {msg.cars && msg.cars.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full mt-1 animate-slide-up">
            {msg.cars.map((car) => (
              <CarCard
                key={car.listing_id}
                car={car}
                isSaved={savedIds.has(car.listing_id)}
                onToggle={() => onToggleFavorite(car)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function CarCard({
  car,
  isSaved,
  onToggle,
}: {
  car: CarListing;
  isSaved: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="glass-panel overflow-hidden rounded-2xl group hover:border-[#8b5cf6]/50 transition-all hover:-translate-y-0.5 hover:shadow-xl hover:shadow-[#8b5cf6]/10">
      {/* Image */}
      <div className="h-40 w-full bg-gray-900 relative">
        {car.hero_image ? (
          <img
            src={car.hero_image}
            alt={car.title}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <Car className="w-10 h-10 text-gray-700" />
          </div>
        )}

        {/* Similarity badge */}
        {car.similarity !== undefined && (
          <div className="absolute top-2 left-2 px-2 py-0.5 rounded-full text-xs font-medium bg-black/60 backdrop-blur-md text-green-400 border border-green-400/20">
            {Math.round(car.similarity * 100)}% match
          </div>
        )}

        {/* Star */}
        <button
          onClick={onToggle}
          className={`absolute top-2 right-2 p-2 rounded-full backdrop-blur-md transition-colors ${
            isSaved
              ? 'bg-yellow-500/80 text-white hover:bg-yellow-600'
              : 'bg-black/40 text-white hover:text-yellow-400 hover:bg-black/60'
          }`}
          title={isSaved ? 'Remove from favorites' : 'Save to favorites'}
        >
          {isSaved ? <StarOff className="w-4 h-4" /> : <Star className="w-4 h-4" />}
        </button>

        {/* Price */}
        <div className="absolute bottom-2 left-2 px-2.5 py-1 rounded-lg bg-black/70 backdrop-blur-md text-white font-bold text-sm border border-white/10">
          {car.price_display || 'Price N/A'}
        </div>
      </div>

      {/* Info */}
      <div className="p-4">
        <h4 className="font-bold text-white text-sm line-clamp-1 mb-2 group-hover:text-[#a78bfa] transition-colors">
          {car.title}
        </h4>
        <div className="grid grid-cols-3 gap-x-2 text-xs text-gray-400">
          <span>{car.year}</span>
          <span className="text-center">{car.mileage_km ? `${car.mileage_km.toLocaleString()} km` : '—'}</span>
          <span className="text-right truncate">{car.location}</span>
        </div>
        <div className="flex gap-2 mt-2 text-xs text-gray-500">
          <span className="px-2 py-0.5 rounded-full bg-white/5 border border-white/10">{car.fuel_type}</span>
          <span className="px-2 py-0.5 rounded-full bg-white/5 border border-white/10">{car.transmission}</span>
        </div>
      </div>

      {car.listing_url && (
        <div className="px-4 pb-3">
          <a
            href={car.listing_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-[#8b5cf6] hover:text-[#a78bfa] transition-colors"
          >
            View on PakWheels →
          </a>
        </div>
      )}
    </div>
  );
}
