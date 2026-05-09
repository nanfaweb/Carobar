"use client";

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Car, Star, Trash2, Home, ArrowLeft, Loader2 } from 'lucide-react';
import { supabase } from '@/lib/supabase';

type Favorite = {
  id: string;
  listing_id: number;
  listings: {
    listing_id: number;
    title: string;
    price_display: string;
    year: number;
    mileage_km: number;
    location: string;
    hero_image: string;
    make: string;
    model: string;
  };
};

export default function FavoritesPage() {
  const [favorites, setFavorites] = useState<Favorite[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchFavorites();
  }, []);

  const fetchFavorites = async () => {
    setLoading(true);
    try {
      const { data: userData } = await supabase.auth.getUser();
      if (!userData.user) {
        // Not logged in
        setLoading(false);
        return;
      }

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
            location,
            hero_image,
            make,
            model
          )
        `)
        .eq('user_id', userData.user.id)
        .order('created_at', { ascending: false });

      if (error) {
        console.error('Error fetching favorites:', error);
      } else {
        setFavorites(data as unknown as Favorite[]);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const removeFavorite = async (id: string) => {
    try {
      const { error } = await supabase.from('favorites').delete().eq('id', id);
      if (!error) {
        setFavorites(prev => prev.filter(f => f.id !== id));
      }
    } catch (err) {
      console.error('Failed to remove favorite', err);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-dark-900 relative overflow-hidden">
      {/* Background decoration */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
        <div className="absolute top-[-20%] left-[20%] w-[600px] h-[600px] bg-brand-500/10 rounded-full blur-[120px]"></div>
      </div>

      {/* Navbar */}
      <nav className="w-full glass-panel py-4 px-6 md:px-12 flex justify-between items-center sticky top-0 z-50">
        <div className="flex items-center gap-4">
          <Link href="/chat" className="p-2 rounded-xl bg-white/5 hover:bg-white/10 transition-colors text-gray-300 hover:text-white">
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-yellow-400 to-orange-500 flex items-center justify-center shadow-lg shadow-yellow-500/20">
              <Star className="text-white w-6 h-6" />
            </div>
            <span className="text-xl font-bold text-white">
              Favorites
            </span>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <Link href="/" className="text-sm flex items-center gap-2 text-gray-300 hover:text-white transition-colors glass-panel px-4 py-2 rounded-full">
            <Home className="w-4 h-4" />
            <span className="hidden sm:inline">Home</span>
          </Link>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-grow p-6 md:p-12 max-w-7xl mx-auto w-full z-10">
        
        <div className="mb-8 animate-slide-up">
          <h1 className="text-3xl md:text-4xl font-bold text-white mb-2">Saved Vehicles</h1>
          <p className="text-gray-400">Keep track of the cars you love and compare them later.</p>
        </div>

        {loading ? (
          <div className="flex flex-col items-center justify-center py-20 text-brand-500">
            <Loader2 className="w-10 h-10 animate-spin mb-4" />
            <p className="text-gray-400">Loading your favorites...</p>
          </div>
        ) : favorites.length === 0 ? (
          <div className="glass-panel p-12 rounded-3xl flex flex-col items-center justify-center text-center animate-fade-in border-dashed border-2 border-white/10">
            <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mb-6 text-gray-500">
              <Star className="w-10 h-10" />
            </div>
            <h2 className="text-2xl font-bold text-white mb-2">No favorites yet</h2>
            <p className="text-gray-400 max-w-md mb-8">
              When you see a car you like in the chat, click the star icon to save it here for quick access later.
            </p>
            <Link href="/chat" className="btn-primary px-8 py-3 rounded-xl font-medium inline-flex items-center gap-2">
              <Car className="w-5 h-5" />
              Start Searching
            </Link>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-slide-up" style={{ animationDelay: '0.1s' }}>
            {favorites.map((fav) => (
              <div key={fav.id} className="glass-panel overflow-hidden rounded-2xl group hover:border-brand-500/50 transition-all hover:shadow-2xl hover:shadow-brand-500/10 hover:-translate-y-1">
                <div className="h-52 w-full bg-gray-800 relative">
                  {fav.listings?.hero_image ? (
                    <img src={fav.listings.hero_image} alt={fav.listings.title} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700" />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <Car className="w-12 h-12 text-gray-600" />
                    </div>
                  )}
                  <div className="absolute top-3 right-3">
                    <button 
                      onClick={() => removeFavorite(fav.id)}
                      className="p-2.5 rounded-xl bg-red-500/80 backdrop-blur-md text-white hover:bg-red-600 transition-colors shadow-lg"
                      title="Remove from favorites"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                  <div className="absolute bottom-3 left-3 px-3 py-1.5 rounded-lg bg-black/70 backdrop-blur-md text-white font-bold text-lg border border-white/10 shadow-lg">
                    {fav.listings?.price_display || 'Price N/A'}
                  </div>
                </div>
                <div className="p-5">
                  <h3 className="font-bold text-white text-lg line-clamp-1 mb-2 group-hover:text-brand-400 transition-colors">
                    {fav.listings?.title}
                  </h3>
                  <div className="grid grid-cols-2 gap-y-2 text-sm text-gray-400">
                    <div className="flex flex-col">
                      <span className="text-xs text-gray-500">Year</span>
                      <span className="text-gray-200">{fav.listings?.year || 'N/A'}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-xs text-gray-500">Mileage</span>
                      <span className="text-gray-200">{fav.listings?.mileage_km ? `${fav.listings.mileage_km.toLocaleString()} km` : 'N/A'}</span>
                    </div>
                    <div className="flex flex-col col-span-2 mt-1 pt-2 border-t border-white/5">
                      <span className="text-xs text-gray-500">Location</span>
                      <span className="text-gray-200">{fav.listings?.location || 'N/A'}</span>
                    </div>
                  </div>
                </div>
                <div className="p-3 bg-white/5 border-t border-white/5 flex">
                  <a 
                    href={`/listing/${fav.listing_id}`} 
                    className="w-full text-center text-sm font-medium text-brand-400 hover:text-brand-300 py-1.5 transition-colors"
                  >
                    View Details
                  </a>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
