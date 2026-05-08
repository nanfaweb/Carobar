'use client';

import React, { useState } from 'react';
import { Search, MapPin, Gauge, Fuel, Calendar, Zap, ChevronRight } from 'lucide-react';

export default function Home() {
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    setIsSearching(true);
    // Simulate RAG API Call
    setTimeout(() => {
      setIsSearching(false);
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-[#09090b] text-white overflow-hidden selection:bg-rose-500/30">
      {/* Dynamic Background Elements */}
      <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-[20%] -left-[10%] w-[50%] h-[50%] rounded-full bg-rose-600/20 blur-[120px] opacity-50 mix-blend-screen animate-pulse duration-10000" />
        <div className="absolute top-[20%] -right-[10%] w-[40%] h-[40%] rounded-full bg-violet-600/20 blur-[120px] opacity-40 mix-blend-screen" />
        <div className="absolute bottom-[-10%] left-[20%] w-[60%] h-[60%] rounded-full bg-blue-600/10 blur-[150px] opacity-30 mix-blend-screen" />
      </div>

      {/* Navbar */}
      <nav className="relative z-10 border-b border-white/5 backdrop-blur-md bg-black/20">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500 to-violet-600 flex items-center justify-center shadow-lg shadow-rose-500/20">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <span className="text-2xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-white/70">
              CAROBAR
            </span>
          </div>
          <div className="hidden md:flex items-center gap-8 text-sm font-medium text-white/60">
            <a href="#" className="hover:text-white transition-colors">Browse</a>
            <a href="#" className="hover:text-white transition-colors">How it Works</a>
            <a href="#" className="hover:text-white transition-colors">About</a>
            <button className="px-5 py-2.5 rounded-full bg-white/10 hover:bg-white/15 text-white transition-all hover:scale-105 active:scale-95 border border-white/10 shadow-xl backdrop-blur-lg">
              Sign In
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 pt-24 pb-32 flex flex-col items-center text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 backdrop-blur-md mb-8 shadow-2xl">
          <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
          <span className="text-sm font-medium text-white/80">AI-Powered Vehicle Intelligence</span>
        </div>

        <h1 className="text-6xl md:text-8xl font-extrabold tracking-tight mb-6 leading-tight">
          Find your perfect car.<br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-rose-400 via-fuchsia-500 to-violet-500">
            No hallucinations.
          </span>
        </h1>

        <p className="text-lg md:text-xl text-white/50 max-w-2xl mb-12 font-light leading-relaxed">
          Ask us anything. Our RAG system analyzes millions of up-to-date Pakistani market listings to give you exact prices, specs, and real recommendations.
        </p>

        {/* Search Interface */}
        <div className="w-full max-w-3xl relative group">
          <div className="absolute -inset-1 bg-gradient-to-r from-rose-500 to-violet-500 rounded-[2rem] blur opacity-25 group-hover:opacity-40 transition duration-1000 group-hover:duration-200" />
          <form 
            onSubmit={handleSearch}
            className="relative flex items-center w-full bg-[#111113]/90 backdrop-blur-xl border border-white/10 rounded-[2rem] p-2 shadow-2xl transition-all"
          >
            <div className="pl-6 pr-4 flex items-center justify-center text-white/40">
              <Search className="w-6 h-6" />
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="e.g., 'Find me a Honda Civic under 50 lakh in Lahore...'"
              className="w-full bg-transparent border-none outline-none text-lg text-white placeholder:text-white/30 h-16 py-4"
            />
            <button 
              type="submit"
              disabled={isSearching}
              className="h-14 px-8 rounded-full bg-white text-black font-semibold hover:bg-gray-100 transition-all flex items-center gap-2 group/btn disabled:opacity-70"
            >
              {isSearching ? (
                <div className="w-5 h-5 border-2 border-black/20 border-t-black rounded-full animate-spin" />
              ) : (
                <>
                  Search
                  <ChevronRight className="w-4 h-4 group-hover/btn:translate-x-1 transition-transform" />
                </>
              )}
            </button>
          </form>
        </div>

        {/* Suggested Queries */}
        <div className="mt-10 flex flex-wrap items-center justify-center gap-3">
          <span className="text-sm text-white/40 mr-2">Try asking:</span>
          {['Toyota Corolla 2021 fuel average?', 'SUVs in Islamabad under 8M', 'Hybrid cars with less than 50k mileage'].map((q) => (
            <button 
              key={q}
              onClick={() => setSearchQuery(q)}
              className="text-sm px-4 py-2 rounded-full border border-white/10 bg-white/5 hover:bg-white/10 text-white/70 hover:text-white transition-colors"
            >
              {q}
            </button>
          ))}
        </div>
      </main>

      {/* Feature Grid */}
      <section className="relative z-10 border-t border-white/5 bg-black/40 backdrop-blur-lg pt-24 pb-32">
        <div className="max-w-7xl mx-auto px-6 grid grid-cols-1 md:grid-cols-3 gap-8">
          {[
            { icon: MapPin, title: 'Hyper-Local Data', desc: 'Real-time listings from Karachi to Peshawar, with accurate location verification.' },
            { icon: Gauge, title: 'Verified Specs', desc: 'No more guessing. We cross-reference engine CC, mileage, and condition automatically.' },
            { icon: Search, title: 'Semantic Search', desc: 'Powered by pgvector. Search by feeling, use-case, or budget—not just keywords.' }
          ].map((feature, idx) => (
            <div key={idx} className="p-8 rounded-3xl bg-white/5 border border-white/5 hover:bg-white/10 transition-all hover:-translate-y-1 duration-300 group">
              <div className="w-14 h-14 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300 shadow-xl">
                <feature.icon className="w-6 h-6 text-rose-400" />
              </div>
              <h3 className="text-xl font-bold mb-3 text-white/90">{feature.title}</h3>
              <p className="text-white/50 leading-relaxed">{feature.desc}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
