import Link from 'next/link';
import { Car, Search, MessageSquare, Star, ArrowRight } from 'lucide-react';

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden">
      {/* Navbar */}
      <nav className="w-full glass-panel py-4 px-6 md:px-12 flex justify-between items-center sticky top-0 z-50">
        <div className="flex items-center gap-2">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-500 to-blue-500 flex items-center justify-center shadow-lg shadow-brand-500/30">
            <Car className="text-white w-6 h-6" />
          </div>
          <span className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">
            Carobar
          </span>
        </div>
        <div className="flex items-center gap-4">
          <Link href="/login" className="text-sm text-gray-300 hover:text-white transition-colors">
            Sign In
          </Link>
          <Link href="/signup" className="text-sm px-5 py-2.5 rounded-full btn-primary font-medium">
            Get Started
          </Link>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-grow flex flex-col items-center justify-center p-6 text-center z-10 relative mt-12 md:mt-0">
        
        {/* Hero Badge */}
        <div className="animate-fade-in inline-flex items-center gap-2 px-4 py-2 rounded-full glass-panel mb-8 border border-brand-500/30 text-brand-500 text-sm font-medium">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-500 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-brand-500"></span>
          </span>
          Powered by Advanced RAG AI
        </div>

        <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6 animate-slide-up max-w-4xl">
          Find your dream car with <br className="hidden md:block" />
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-brand-500 to-blue-400">
            Intelligence
          </span>
        </h1>
        
        <p className="text-lg md:text-xl text-gray-400 max-w-2xl mb-12 animate-slide-up" style={{ animationDelay: '0.1s' }}>
          Experience the future of automotive search. Ask our AI anything about cars, get real-time market data, and discover perfect matches instantly without the hassle.
        </p>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 mb-20 animate-slide-up" style={{ animationDelay: '0.2s' }}>
          <Link href="/chat" className="flex items-center justify-center gap-2 px-8 py-4 rounded-full btn-primary text-lg font-medium shadow-xl shadow-brand-500/20 group">
            <MessageSquare className="w-5 h-5" />
            Start Chatting
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </Link>
          <Link href="/favorites" className="flex items-center justify-center gap-2 px-8 py-4 rounded-full btn-secondary text-lg font-medium glass-panel">
            <Star className="w-5 h-5 text-yellow-400" />
            View Favorites
          </Link>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-6xl w-full px-4 animate-slide-up" style={{ animationDelay: '0.3s' }}>
          <FeatureCard 
            icon={<Search className="w-6 h-6 text-brand-500" />}
            title="Semantic Search"
            description="Find cars by describing what you need. Our AI understands context, not just keywords."
          />
          <FeatureCard 
            icon={<MessageSquare className="w-6 h-6 text-blue-400" />}
            title="Interactive Assistant"
            description="Ask follow-up questions, compare models, and get expert advice in real-time."
          />
          <FeatureCard 
            icon={<Car className="w-6 h-6 text-purple-400" />}
            title="Live Market Data"
            description="Get accurate, up-to-date pricing and specifications from actual listings."
          />
        </div>
      </main>

      {/* Background decoration */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-brand-500/10 rounded-full blur-[120px] -z-10 pointer-events-none"></div>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) {
  return (
    <div className="glass-panel p-8 rounded-2xl text-left hover:border-brand-500/50 transition-colors group cursor-default">
      <div className="w-12 h-12 rounded-xl bg-white/5 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
        {icon}
      </div>
      <h3 className="text-xl font-bold mb-3 text-white">{title}</h3>
      <p className="text-gray-400 leading-relaxed">
        {description}
      </p>
    </div>
  );
}
