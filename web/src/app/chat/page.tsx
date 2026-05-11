"use client";

import { useState, useRef, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  Car, Send, Star, StarOff, User, Bot,
  LogOut, Home, Menu, X, MessageSquare, AlertCircle, Sparkles,
  TrendingUp, TrendingDown, Minus, Copy, Check, Brain, Loader2, UserCircle2, ChevronLeft
} from 'lucide-react';

import { supabase } from '@/lib/supabase';
import ReactMarkdown from 'react-markdown';
import type { User as SupabaseUser } from '@supabase/supabase-js';

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

type PriceAnalysisResult = {
  verdict: 'Great Deal' | 'Fair Price' | 'Overpriced';
  avg_similar_price: number;
  similar_count: number;
  price_difference: number;
  price_difference_pct: number;
  analysis: string;
  negotiation_message: string;
};

type RecommendedCar = CarListing & { why_match: string };

// ─── Login Modal ──────────────────────────────────────────────────────────────

function LoginModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
      <div
        className="relative glass-panel rounded-3xl p-8 max-w-sm w-full text-center animate-slide-up shadow-2xl shadow-[#8b5cf6]/20 border border-[#8b5cf6]/20"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 rounded-full text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>

        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#8b5cf6] to-blue-500 flex items-center justify-center mx-auto mb-5 shadow-lg shadow-[#8b5cf6]/30">
          <Star className="w-8 h-8 text-white" />
        </div>

        <h2 className="text-2xl font-bold text-white mb-2">Save Your Favourites</h2>
        <p className="text-gray-400 text-sm mb-7 leading-relaxed">
          Sign in to save cars, pick up your chats where you left off, and access your garage from any device.
        </p>

        <div className="flex flex-col gap-3">
          <Link
            href="/login"
            className="btn-primary py-3 rounded-xl font-semibold flex items-center justify-center gap-2"
          >
            Sign In
          </Link>
          <Link
            href="/signup"
            className="btn-secondary py-3 rounded-xl font-semibold flex items-center justify-center gap-2"
          >
            Create Account
          </Link>
        </div>
      </div>
    </div>
  );
}

// ─── Component ───────────────────────────────────────────────────────────────

export default function ChatPage() {
  const WELCOME: Message = {
    id: 'welcome-message',

    role: 'assistant',
    content:
      'Hi! I\'m your Carobar AI assistant, powered by real car listings. Try asking me:\n\n• "Show me Honda Civics under PKR 5 million"\n• "Best automatic cars in Lahore"\n• "Compare Toyota Corolla vs Honda City"',
  };

  const [messages, setMessages] = useState<Message[]>([WELCOME]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [savedIds, setSavedIds] = useState<Set<number>>(new Set());
  const [user, setUser] = useState<SupabaseUser | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [showLoginModal, setShowLoginModal] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  const [analyzerCar,     setAnalyzerCar]     = useState<CarListing | null>(null);
  const [analyzerLoading, setAnalyzerLoading] = useState(false);
  const [analyzerResult,  setAnalyzerResult]  = useState<PriceAnalysisResult | null>(null);
  const [showNegotiation, setShowNegotiation] = useState(false);
  const [copied,          setCopied]          = useState(false);
  const [recommendations, setRecommendations] = useState<RecommendedCar[]>([]);
  const [recsLoading,     setRecsLoading]     = useState(false);
  const [hasProfile,      setHasProfile]      = useState(false);

  // ── Auth + initial data load ────────────────────────────────────────────

  useEffect(() => {
    const init = async () => {
      const { data: { user: currentUser } } = await supabase.auth.getUser();
      setUser(currentUser);
      setAuthLoading(false);

      if (currentUser) {
        await Promise.all([
          loadConversation(currentUser.id),
          loadFavoriteIds(currentUser.id),
          loadRecommendations(currentUser.id),
        ]);
      }
    };

    init();

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });

    return () => subscription.unsubscribe();
  }, []);

  // ── Scroll to bottom ────────────────────────────────────────────────────

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ── Load / save conversation ────────────────────────────────────────────

  const loadConversation = async (userId: string) => {
    try {
      const { data } = await supabase
        .from('conversations')
        .select('messages')
        .eq('user_id', userId)
        .single();

      if (data?.messages && Array.isArray(data.messages) && data.messages.length > 0) {
        // Restore saved messages, but regenerate IDs so React keys are stable
        const restored = (data.messages as Message[]).map((m, i) => ({ ...m, id: `saved-${i}-${Date.now()}` }));
        setMessages([WELCOME, ...restored]);

      }
    } catch {
      // No existing conversation — that's fine
    }
  };

  const saveConversation = useCallback(async (msgs: Message[], userId: string) => {
    // Don't save the static greeting
    const toSave = msgs.slice(1).map(({ id: _id, ...rest }) => rest);
    if (toSave.length === 0) return;

    await supabase.from('conversations').upsert(
      { user_id: userId, messages: toSave, updated_at: new Date().toISOString() },
      { onConflict: 'user_id' }
    );
  }, []);

  // ── Favorites helpers ───────────────────────────────────────────────────

  const loadFavoriteIds = async (userId: string) => {
    const { data } = await supabase
      .from('favorites')
      .select('listing_id')
      .eq('user_id', userId);
    if (data) setSavedIds(new Set(data.map((r: any) => r.listing_id)));
  };

  const loadRecommendations = async (userId: string) => {
    const { data: profile } = await supabase
      .from('user_profiles').select('*').eq('user_id', userId).single();
    if (!profile) return;
    setHasProfile(true);
    setRecsLoading(true);
    try {
      const res = await fetch('/api/recommendations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile: {
          budget_max: profile.budget_max,
          commute_km: profile.commute_km,
          family_size: profile.family_size,
          fuel_preference: profile.fuel_preference,
          transmission_preference: profile.transmission_preference,
          priorities: profile.priorities || [],
        }}),
      });
      if (res.ok) { const d = await res.json(); setRecommendations(d.recommendations || []); }
    } finally { setRecsLoading(false); }
  };

  const handleAnalyzePrice = async (car: CarListing) => {
    setAnalyzerCar(car);
    setAnalyzerResult(null);
    setShowNegotiation(false);
    setCopied(false);
    setAnalyzerLoading(true);
    try {
      const res = await fetch('/api/analyze-price', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          listing_id: car.listing_id, title: car.title, make: car.make,
          model: car.model, year: car.year, price_pkr: car.price_pkr,
          mileage_km: car.mileage_km, fuel_type: car.fuel_type,
          transmission: car.transmission, location: car.location || '',
        }),
      });
      if (res.ok) setAnalyzerResult(await res.json());
    } finally { setAnalyzerLoading(false); }
  };

  // ── Auth actions ────────────────────────────────────────────────────────

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    setUser(null);
    setSavedIds(new Set());
    setMessages([WELCOME]);
    router.push('/login');
  };

  // ── Chat history ─────────────────────────────────────────────────────

  const buildHistory = (msgs: Message[]): HistoryEntry[] =>
    msgs
      .slice(1)
      .filter((m) => !m.error)
      .map((m) => ({ role: m.role, content: m.content }));

  // ── Send message ────────────────────────────────────────────────────────

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userContent = input.trim();
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: userContent };
    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);
    setInput('');
    setIsLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userContent, history: buildHistory(nextMessages.slice(0, -1)) }),
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

      const finalMessages = [...nextMessages, botMsg];
      setMessages(finalMessages);

      // Persist conversation for logged-in users
      if (user) {
        saveConversation(finalMessages, user.id);
      }
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `Sorry, I ran into a problem: ${err.message}. Make sure the RAG backend is running.`,
          error: true,
        },
      ]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  // ── Toggle favourite ─────────────────────────────────────────────────────

  const toggleFavorite = async (car: CarListing) => {
    if (!user) {
      setShowLoginModal(true);
      return;
    }

    const isSaved = savedIds.has(car.listing_id);

    if (isSaved) {
      const { error } = await supabase
        .from('favorites')
        .delete()
        .eq('user_id', user.id)
        .eq('listing_id', car.listing_id);
      if (!error) setSavedIds((prev) => { const next = new Set(prev); next.delete(car.listing_id); return next; });
      else console.error('[favorites delete]', error?.message, error?.code, error?.details);
    } else {
      const { error } = await supabase
        .from('favorites')
        .insert({ user_id: user.id, listing_id: Number(car.listing_id) });
      if (!error || error.code === '23505') {
        setSavedIds((prev) => new Set([...prev, car.listing_id]));
      } else {
        console.error('[favorites insert] message:', error?.message);
        console.error('[favorites insert] code:', error?.code);
        console.error('[favorites insert] details:', error?.details);
        console.error('[favorites insert] hint:', error?.hint);
      }
    }
  };

  // ── Clear chat ──────────────────────────────────────────────────────────

  const clearChat = async () => {
    setMessages([WELCOME]);
    if (user) {
      await supabase.from('conversations').delete().eq('user_id', user.id);
    }
  };

  // ── Render ──────────────────────────────────────────────────────────────

  return (
    <>
      {showLoginModal && <LoginModal onClose={() => setShowLoginModal(false)} />}
      {analyzerCar && (
        <PriceAnalyzerModal
          car={analyzerCar}
          loading={analyzerLoading}
          result={analyzerResult}
          showNegotiation={showNegotiation}
          setShowNegotiation={setShowNegotiation}
          copied={copied}
          setCopied={setCopied}
          onClose={() => setAnalyzerCar(null)}
        />
      )}

      <div className="flex h-screen overflow-hidden bg-[#050505] text-white">
        {/* Mobile Sidebar Overlay */}
        {isSidebarOpen && (
          <div
            className="fixed inset-0 bg-black/60 z-40 md:hidden backdrop-blur-sm"
            onClick={() => setIsSidebarOpen(false)}
          />
        )}

        {/* ── Sidebar ── */}
        <aside
          className={`fixed inset-y-0 left-0 z-50 w-64 glass-panel border-r border-white/10 transform transition-transform duration-300 ease-in-out
            md:relative md:translate-x-0 flex flex-col ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}
        >
          {/* Logo */}
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

          {/* Nav */}
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
            <Link href="/profile" className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-gray-400 hover:bg-white/5 hover:text-white transition-colors">
              <Brain className="w-5 h-5" />
              My Profile
              {hasProfile && <span className="ml-auto w-2 h-2 rounded-full bg-[#8b5cf6]" />}
            </Link>

            {/* Clear chat */}
            {messages.length > 1 && (
              <button
                onClick={clearChat}
                className="flex items-center gap-3 px-3 py-2.5 w-full rounded-xl text-gray-500 hover:bg-white/5 hover:text-gray-300 transition-colors text-sm mt-4"
              >
                <X className="w-4 h-4" />
                Clear Chat
              </button>
            )}
          </nav>

          {/* User / Auth */}
          <div className="p-4 border-t border-white/10 shrink-0">
            {!authLoading && (
              user ? (
                <div className="space-y-1">
                  <div className="px-3 py-2 rounded-xl bg-white/5 text-sm text-gray-300 truncate">
                    {user.email}
                  </div>
                  <button
                    onClick={handleSignOut}
                    className="flex items-center gap-3 px-3 py-2.5 w-full rounded-xl text-gray-400 hover:bg-white/5 hover:text-red-400 transition-colors"
                  >
                    <LogOut className="w-5 h-5" />
                    Sign Out
                  </button>
                </div>
              ) : (
                <div className="space-y-2">
                  <Link
                    href="/login"
                    className="flex items-center justify-center gap-2 btn-primary px-3 py-2.5 w-full rounded-xl text-sm font-medium"
                  >
                    Sign In
                  </Link>
                  <Link
                    href="/signup"
                    className="flex items-center justify-center gap-2 btn-secondary px-3 py-2.5 w-full rounded-xl text-sm font-medium"
                  >
                    Create Account
                  </Link>
                </div>
              )
            )}
          </div>
        </aside>

        {/* ── Chat Area ── */}
        <main className="flex-1 flex flex-col min-w-0 h-full">
          {/* Header */}
          <header className="h-16 glass-panel border-b border-white/10 flex items-center px-4 shrink-0 z-10 justify-between">
            <div className="flex items-center gap-3">
              <button 
                onClick={() => router.back()}
                className="p-2 text-gray-400 hover:text-white rounded-xl hover:bg-white/5 transition-colors group flex items-center gap-1"
                title="Go back"
              >
                <ChevronLeft className="w-5 h-5 group-hover:-translate-x-0.5 transition-transform" />
                <span className="text-xs font-medium hidden sm:inline">Back</span>
              </button>
              
              <div className="flex items-center gap-3 md:hidden">
                <button onClick={() => setIsSidebarOpen(true)} className="p-2 text-gray-400 hover:text-white rounded-xl hover:bg-white/5">
                  <Menu className="w-6 h-6" />
                </button>
                <span className="font-bold text-white">Carobar</span>
              </div>
            </div>
            
            <div className="hidden md:flex items-center gap-2 text-gray-400 text-sm">

              {user ? (
                <span className="text-gray-500">Chatting as <span className="text-gray-300">{user.email}</span></span>
              ) : (
                <span className="flex items-center gap-2 text-gray-500">
                  <Sparkles className="w-4 h-4" />
                  Sign in to save chats &amp; favourites
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <span className="hidden sm:inline">RAG AI Active</span>
              <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            </div>
          </header>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
            {(hasProfile || recsLoading) && (
              <ProfileRecommendationsBanner
                loading={recsLoading}
                cars={recommendations}
                savedIds={savedIds}
                onToggleFavorite={toggleFavorite}
                onAnalyze={handleAnalyzePrice}
              />
            )}
            {messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                msg={msg}
                savedIds={savedIds}
                onToggleFavorite={toggleFavorite}
                onAnalyze={handleAnalyzePrice}
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
    </>
  );
}

// ─── MessageBubble ────────────────────────────────────────────────────────────

function MessageBubble({
  msg,
  savedIds,
  onToggleFavorite,
  onAnalyze,
}: {
  msg: Message;
  savedIds: Set<number>;
  onToggleFavorite: (car: CarListing) => void;
  onAnalyze: (car: CarListing) => void;
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
          className={`px-5 py-3.5 rounded-2xl leading-relaxed text-sm ${
            isUser
              ? 'bg-[#7c3aed] text-white rounded-tr-sm'
              : msg.error
              ? 'bg-red-500/10 border border-red-500/20 text-red-300 rounded-tl-sm flex items-start gap-2'
              : 'glass-panel text-gray-200 rounded-tl-sm'
          }`}
        >
          {msg.error && <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />}
          <ReactMarkdown
            components={{
              img: ({ node, ...props }) => (
                <img {...props} className="max-w-full rounded-xl my-2 max-h-64 object-cover shadow-lg" />
              ),
              p: ({ node, ...props }) => <p className="mb-2 last:mb-0" {...props} />,
              a: ({ node, ...props }) => <a className="text-[#a78bfa] hover:underline" target="_blank" rel="noopener noreferrer" {...props} />,
              strong: ({ node, ...props }) => <strong className="text-white font-semibold" {...props} />,
            }}
          >
            {msg.content}
          </ReactMarkdown>
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
                onAnalyze={() => onAnalyze(car)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── CarCard ──────────────────────────────────────────────────────────────────

function CarCard({
  car,
  isSaved,
  onToggle,
  onAnalyze,
}: {
  car: CarListing;
  isSaved: boolean;
  onToggle: () => void;
  onAnalyze?: () => void;
}) {
  return (
    <div className="glass-panel overflow-hidden rounded-2xl group hover:border-[#8b5cf6]/50 transition-all hover:-translate-y-0.5 hover:shadow-xl hover:shadow-[#8b5cf6]/10">
      {/* Image */}
      <div className="h-44 w-full bg-gray-900 relative overflow-hidden">
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

        {/* Star / Favourite button */}
        <button
          onClick={onToggle}
          title={isSaved ? 'Remove from saved cars' : 'Save this car'}
          className={`absolute top-2 right-2 p-2 rounded-full backdrop-blur-md transition-all ${
            isSaved
              ? 'bg-yellow-500/90 text-white hover:bg-yellow-600 shadow-lg shadow-yellow-500/30'
              : 'bg-black/50 text-white hover:text-yellow-400 hover:bg-black/70'
          }`}
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
        <div className="grid grid-cols-3 gap-x-2 text-xs text-gray-400 mb-3">
          <span>{car.year}</span>
          <span className="text-center">{car.mileage_km ? `${car.mileage_km.toLocaleString()} km` : '—'}</span>
          <span className="text-right truncate">{car.location}</span>
        </div>
        <div className="flex gap-2 text-xs text-gray-500">
          <span className="px-2 py-0.5 rounded-full bg-white/5 border border-white/10">{car.fuel_type}</span>
          <span className="px-2 py-0.5 rounded-full bg-white/5 border border-white/10">{car.transmission}</span>
        </div>
      </div>

      <div className="px-4 pb-4 flex items-center justify-between gap-2">
        {car.listing_url ? (
          <a href={car.listing_url} target="_blank" rel="noopener noreferrer"
            className="text-xs text-[#8b5cf6] hover:text-[#a78bfa] transition-colors">
            View on PakWheels →
          </a>
        ) : <span />}
        {onAnalyze && (
          <button onClick={onAnalyze}
            className="flex items-center gap-1 text-xs px-3 py-1.5 rounded-full bg-[#8b5cf6]/15 border border-[#8b5cf6]/30 text-[#a78bfa] hover:bg-[#8b5cf6]/30 transition-all font-medium">
            <TrendingUp className="w-3 h-3" /> Analyze Price
          </button>
        )}
      </div>
    </div>
  );
}

// ─── PriceAnalyzerModal ───────────────────────────────────────────────────────

function PriceAnalyzerModal({ car, loading, result, showNegotiation, setShowNegotiation, copied, setCopied, onClose }: {
  car: CarListing; loading: boolean; result: PriceAnalysisResult | null;
  showNegotiation: boolean; setShowNegotiation: (v: boolean) => void;
  copied: boolean; setCopied: (v: boolean) => void; onClose: () => void;
}) {
  const verdictStyles = {
    'Great Deal': { color: 'text-green-400', bg: 'bg-green-400/10 border-green-400/30', Icon: TrendingDown },
    'Fair Price': { color: 'text-yellow-400', bg: 'bg-yellow-400/10 border-yellow-400/30', Icon: Minus },
    'Overpriced': { color: 'text-red-400',   bg: 'bg-red-400/10   border-red-400/30',   Icon: TrendingUp  },
  };
  const style = result ? verdictStyles[result.verdict] : null;

  const copyMsg = () => {
    if (!result) return;
    navigator.clipboard.writeText(result.negotiation_message);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />
      <div className="relative glass-panel rounded-3xl p-6 max-w-md w-full max-h-[90vh] overflow-y-auto no-scrollbar animate-slide-up border border-[#8b5cf6]/20 shadow-2xl shadow-[#8b5cf6]/20"
        onClick={e => e.stopPropagation()}>


        <button onClick={onClose} className="absolute top-4 right-4 p-2 rounded-full text-gray-400 hover:text-white hover:bg-white/10">
          <X className="w-4 h-4" />
        </button>

        <div className="flex items-center gap-2 mb-1">
          <TrendingUp className="w-5 h-5 text-[#8b5cf6]" />
          <h2 className="font-bold text-white text-lg">Price Analysis</h2>
        </div>
        <p className="text-gray-400 text-sm mb-4 line-clamp-1">{car.title}</p>

        {loading && (
          <div className="flex flex-col items-center py-10 gap-3 text-gray-400">
            <Loader2 className="w-8 h-8 animate-spin text-[#8b5cf6]" />
            <p className="text-sm">Analyzing market data…</p>
          </div>
        )}

        {!loading && result && style && (
          <>
            {/* Verdict badge */}
            <div className={`flex items-center gap-3 p-4 rounded-2xl border mb-4 ${style.bg}`}>
              <style.Icon className={`w-6 h-6 ${style.color} shrink-0`} />
              <div>
                <p className={`font-bold text-xl ${style.color}`}>{result.verdict}</p>
                <p className="text-gray-400 text-xs">
                  {result.similar_count > 0
                    ? `Based on ${result.similar_count} similar listing${result.similar_count > 1 ? 's' : ''}`
                    : 'Limited comparison data available'}
                </p>
              </div>
            </div>

            {/* Price comparison */}
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="glass-panel rounded-xl p-3">
                <p className="text-xs text-gray-500 mb-1">Asking Price</p>
                <p className="font-bold text-white text-sm">{car.price_display}</p>
              </div>
              <div className="glass-panel rounded-xl p-3">
                <p className="text-xs text-gray-500 mb-1">Market Average</p>
                <p className="font-bold text-white text-sm">PKR {result.avg_similar_price.toLocaleString()}</p>
              </div>
            </div>

            {/* Difference pill */}
            {result.similar_count > 0 && (
              <div className={`text-xs font-medium px-3 py-1.5 rounded-full inline-flex items-center gap-1 mb-4 ${style.bg} ${style.color} border`}>
                <style.Icon className="w-3 h-3" />
                {result.price_difference > 0
                  ? `PKR ${Math.abs(result.price_difference).toLocaleString()} above avg (${result.price_difference_pct}%)`
                  : `PKR ${Math.abs(result.price_difference).toLocaleString()} below avg (${Math.abs(result.price_difference_pct)}%)`}
              </div>
            )}

            {/* AI analysis */}
            <p className="text-sm text-gray-300 mb-4 leading-relaxed">{result.analysis}</p>

            {/* Negotiation message */}
            <button onClick={() => setShowNegotiation(!showNegotiation)}
              className="w-full flex items-center justify-between px-4 py-3 rounded-xl bg-white/5 border border-white/10 hover:border-[#8b5cf6]/40 transition-all text-sm font-medium text-gray-300 mb-2">
              <span>💬 Draft Negotiation Message</span>
              {showNegotiation ? <TrendingDown className="w-4 h-4" /> : <TrendingUp className="w-4 h-4" />}
            </button>

            {showNegotiation && (
              <div className="bg-white/5 border border-white/10 rounded-xl p-4 animate-slide-up">
                <p className="text-sm text-gray-300 leading-relaxed mb-3">{result.negotiation_message}</p>
                <button onClick={copyMsg}
                  className={`flex items-center gap-2 text-xs px-4 py-2 rounded-full font-medium transition-all ${
                    copied ? 'bg-green-500/20 border border-green-500/30 text-green-400' : 'bg-[#8b5cf6]/20 border border-[#8b5cf6]/30 text-[#a78bfa] hover:bg-[#8b5cf6]/30'
                  }`}>
                  {copied ? <><Check className="w-3 h-3" /> Copied!</> : <><Copy className="w-3 h-3" /> Copy Message</>}
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ─── ProfileRecommendationsBanner ────────────────────────────────────────────

function ProfileRecommendationsBanner({ loading, cars, savedIds, onToggleFavorite, onAnalyze }: {
  loading: boolean; cars: (CarListing & { why_match: string })[];
  savedIds: Set<number>; onToggleFavorite: (car: CarListing) => void;
  onAnalyze: (car: CarListing) => void;
}) {
  return (
    <div className="max-w-4xl mx-auto glass-panel rounded-2xl border border-[#8b5cf6]/25 overflow-hidden animate-slide-up">
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/10">
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-[#8b5cf6]" />
          <span className="text-sm font-semibold text-white">Your Personal Matches</span>
          <span className="text-xs text-gray-500">based on your profile</span>
        </div>
        <a href="/profile" className="text-xs text-[#8b5cf6] hover:text-[#a78bfa] transition-colors flex items-center gap-1">
          <UserCircle2 className="w-3 h-3" /> Update Profile
        </a>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-8 gap-3 text-gray-500">
          <Loader2 className="w-5 h-5 animate-spin text-[#8b5cf6]" />
          <span className="text-sm">Finding your perfect matches…</span>
        </div>
      ) : cars.length === 0 ? (
        <p className="text-center text-sm text-gray-500 py-6">No matches found — try updating your profile.</p>
      ) : (
        <div className="flex gap-4 overflow-x-auto p-4 pb-5 scrollbar-thin">
          {cars.map(car => {
            const saved = savedIds.has(car.listing_id);
            return (
              <div key={car.listing_id}
                className="glass-panel rounded-xl overflow-hidden border border-white/10 hover:border-[#8b5cf6]/40 transition-all shrink-0 w-56">
                <div className="h-32 relative overflow-hidden bg-gray-900">
                  {car.hero_image
                    ? <img src={car.hero_image} alt={car.title} className="w-full h-full object-cover" />
                    : <div className="w-full h-full flex items-center justify-center"><Car className="w-8 h-8 text-gray-700" /></div>}
                  <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded-lg bg-black/70 text-white text-xs font-bold border border-white/10">
                    {car.price_display}
                  </div>
                  <button onClick={() => onToggleFavorite(car)}
                    className={`absolute top-2 right-2 p-1.5 rounded-full backdrop-blur-md transition-all ${
                      saved ? 'bg-yellow-500/90 text-white' : 'bg-black/50 text-white hover:text-yellow-400'
                    }`}>
                    {saved ? <StarOff className="w-3 h-3" /> : <Star className="w-3 h-3" />}
                  </button>
                </div>
                <div className="p-3">
                  <p className="text-white text-xs font-bold line-clamp-1 mb-1">{car.title}</p>
                  <p className="text-gray-400 text-xs leading-snug line-clamp-2 mb-2">{car.why_match}</p>
                  <button onClick={() => onAnalyze(car)}
                    className="text-xs text-[#a78bfa] hover:text-[#8b5cf6] transition-colors flex items-center gap-1">
                    <TrendingUp className="w-3 h-3" /> Analyze Price
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
