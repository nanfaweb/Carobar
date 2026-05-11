"use client";

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Car, Star, Trash2, Home, ArrowLeft, Loader2, LogOut, MessageSquare, Lock, ChevronLeft } from 'lucide-react';

import { supabase } from '@/lib/supabase';
import type { User } from '@supabase/supabase-js';

type Favorite = {
  id: string;
  listing_id: number;
  listings: {
    listing_id: number;
    title: string;
    price_display: string;
    year: number;
    mileage_km: number;
    fuel_type: string;
    transmission: string;
    location: string;
    hero_image: string;
    listing_url: string;
    make: string;
    model: string;
  };
};

export default function FavoritesPage() {
  const [favorites, setFavorites] = useState<Favorite[]>([]);
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState<User | null>(null);
  const router = useRouter();

  useEffect(() => {
    const init = async () => {
      const { data: { user: currentUser } } = await supabase.auth.getUser();
      setUser(currentUser);
      if (currentUser) {
        await fetchFavorites(currentUser.id);
      }
      setLoading(false);
    };
    init();
  }, []);

  const fetchFavorites = async (userId: string) => {
    try {
      const { data, error } = await supabase
        .from('favorites')
        .select(`
          id,
          listing_id,
          listings (
            listing_id,
            title,
            price_display,
            year,
            mileage_km,
            fuel_type,
            transmission,
            location,
            hero_image,
            listing_url,
            make,
            model
          )
        `)
        .eq('user_id', userId)
        .order('created_at', { ascending: false });

      if (!error) setFavorites(data as unknown as Favorite[]);
    } catch (err) {
      console.error(err);
    }
  };

  const removeFavorite = async (id: string) => {
    const { error } = await supabase.from('favorites').delete().eq('id', id);
    if (!error) setFavorites(prev => prev.filter(f => f.id !== id));
  };

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    router.push('/login');
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#050505] relative overflow-hidden">
      {/* Background glow */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
        <div className="absolute top-[-20%] left-[20%] w-[600px] h-[600px] bg-[#8b5cf6]/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[10%] w-[400px] h-[400px] bg-blue-500/10 rounded-full blur-[120px]" />
      </div>

      {/* Navbar */}
      <nav className="w-full glass-panel py-4 px-6 md:px-12 flex justify-between items-center sticky top-0 z-50">
        <div className="flex items-center gap-4">
          <button 
            onClick={() => router.back()}
            className="p-2 rounded-xl bg-white/5 hover:bg-white/10 transition-colors text-gray-300 hover:text-white flex items-center gap-1 group"
            title="Go back"
          >
            <ChevronLeft className="w-5 h-5 group-hover:-translate-x-0.5 transition-transform" />
            <span className="text-xs font-medium hidden sm:inline pr-1">Back</span>
          </button>

          <div className="flex items-center gap-2">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-yellow-400 to-orange-500 flex items-center justify-center shadow-lg shadow-yellow-500/20">
              <Star className="text-white w-5 h-5" />
            </div>
            <span className="text-xl font-bold text-white">Saved Cars</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Link href="/" className="text-sm flex items-center gap-2 text-gray-300 hover:text-white transition-colors glass-panel px-4 py-2 rounded-full">
            <Home className="w-4 h-4" />
            <span className="hidden sm:inline">Home</span>
          </Link>
          {user && (
            <button
              onClick={handleSignOut}
              className="text-sm flex items-center gap-2 text-gray-400 hover:text-red-400 transition-colors glass-panel px-4 py-2 rounded-full"
            >
              <LogOut className="w-4 h-4" />
              <span className="hidden sm:inline">Sign Out</span>
            </button>
          )}
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-grow p-6 md:p-12 max-w-7xl mx-auto w-full z-10">

        {/* Heading */}
        <div className="mb-8 animate-slide-up">
          <h1 className="text-3xl md:text-4xl font-bold text-white mb-2">Saved Vehicles</h1>
          <p className="text-gray-400">Keep track of the cars you love and compare them later.</p>
        </div>

        {/* Loading */}
        {loading ? (
          <div className="flex flex-col items-center justify-center py-20 text-[#8b5cf6]">
            <Loader2 className="w-10 h-10 animate-spin mb-4" />
            <p className="text-gray-400">Loading your garage...</p>
          </div>

        /* Auth wall — not logged in */
        ) : !user ? (
          <div className="glass-panel p-12 rounded-3xl flex flex-col items-center justify-center text-center animate-fade-in border border-[#8b5cf6]/20 max-w-lg mx-auto">
            <div className="w-20 h-20 rounded-full bg-gradient-to-br from-[#8b5cf6]/20 to-blue-500/20 border border-[#8b5cf6]/30 flex items-center justify-center mb-6">
              <Lock className="w-10 h-10 text-[#8b5cf6]" />
            </div>
            <h2 className="text-2xl font-bold text-white mb-2">Sign In to View Saved Cars</h2>
            <p className="text-gray-400 max-w-sm mb-8 leading-relaxed">
              Your saved cars are tied to your account. Sign in to access your garage and add cars from the AI chat.
            </p>
            <div className="flex flex-col sm:flex-row gap-3 w-full max-w-xs">
              <Link href="/login" className="btn-primary py-3 px-6 rounded-xl font-semibold text-center flex-1">
                Sign In
              </Link>
              <Link href="/signup" className="btn-secondary py-3 px-6 rounded-xl font-semibold text-center flex-1">
                Sign Up
              </Link>
            </div>
          </div>

        /* Empty state */
        ) : favorites.length === 0 ? (
          <div className="glass-panel p-12 rounded-3xl flex flex-col items-center justify-center text-center animate-fade-in border-dashed border-2 border-white/10">
            <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mb-6 text-gray-500">
              <Star className="w-10 h-10" />
            </div>
            <h2 className="text-2xl font-bold text-white mb-2">No saved cars yet</h2>
            <p className="text-gray-400 max-w-md mb-8">
              When you find a car you like in the AI chat, click the ⭐ star on the card to save it here.
            </p>
            <Link href="/chat" className="btn-primary px-8 py-3 rounded-xl font-semibold inline-flex items-center gap-2">
              <MessageSquare className="w-5 h-5" />
              Start Searching
            </Link>
          </div>

        /* Favourites grid */
        ) : (
          <>
            <p className="text-gray-500 text-sm mb-6">{favorites.length} saved {favorites.length === 1 ? 'car' : 'cars'}</p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-slide-up">
              {favorites.map((fav) => (
                <div
                  key={fav.id}
                  className="glass-panel overflow-hidden rounded-2xl group hover:border-[#8b5cf6]/50 transition-all hover:shadow-2xl hover:shadow-[#8b5cf6]/10 hover:-translate-y-1"
                >
                  {/* Image */}
                  <div className="h-52 w-full bg-gray-900 relative overflow-hidden">
                    {fav.listings?.hero_image ? (
                      <img
                        src={fav.listings.hero_image}
                        alt={fav.listings.title}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <Car className="w-12 h-12 text-gray-700" />
                      </div>
                    )}

                    {/* Remove button */}
                    <button
                      onClick={() => removeFavorite(fav.id)}
                      className="absolute top-3 right-3 p-2.5 rounded-xl bg-red-500/80 backdrop-blur-md text-white hover:bg-red-600 transition-colors shadow-lg"
                      title="Remove from saved cars"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>

                    {/* Price badge */}
                    <div className="absolute bottom-3 left-3 px-3 py-1.5 rounded-lg bg-black/70 backdrop-blur-md text-white font-bold text-base border border-white/10 shadow-lg">
                      {fav.listings?.price_display || 'Price N/A'}
                    </div>
                  </div>

                  {/* Details */}
                  <div className="p-5">
                    <h3 className="font-bold text-white text-base line-clamp-1 mb-3 group-hover:text-[#a78bfa] transition-colors">
                      {fav.listings?.title}
                    </h3>
                    <div className="grid grid-cols-2 gap-y-3 text-sm">
                      <div className="flex flex-col">
                        <span className="text-xs text-gray-500 mb-0.5">Year</span>
                        <span className="text-gray-200 font-medium">{fav.listings?.year || 'N/A'}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-xs text-gray-500 mb-0.5">Mileage</span>
                        <span className="text-gray-200 font-medium">
                          {fav.listings?.mileage_km ? `${fav.listings.mileage_km.toLocaleString()} km` : 'N/A'}
                        </span>
                      </div>
                      <div className="flex flex-col col-span-2 pt-2 border-t border-white/5">
                        <span className="text-xs text-gray-500 mb-0.5">Location</span>
                        <span className="text-gray-200 font-medium">{fav.listings?.location || 'N/A'}</span>
                      </div>
                    </div>

                    {/* Tags */}
                    <div className="flex gap-2 mt-3">
                      {fav.listings?.fuel_type && (
                        <span className="px-2 py-0.5 rounded-full text-xs text-gray-400 bg-white/5 border border-white/10">
                          {fav.listings.fuel_type}
                        </span>
                      )}
                      {fav.listings?.transmission && (
                        <span className="px-2 py-0.5 rounded-full text-xs text-gray-400 bg-white/5 border border-white/10">
                          {fav.listings.transmission}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Footer */}
                  {fav.listings?.listing_url && (
                    <div className="px-5 pb-4">
                      <a
                        href={fav.listings.listing_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-[#8b5cf6] hover:text-[#a78bfa] transition-colors"
                      >
                        View on PakWheels →
                      </a>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </main>
    </div>
  );
}
