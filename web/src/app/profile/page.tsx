"use client";

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Brain, ChevronRight, ChevronLeft, Check, Loader2 } from 'lucide-react';
import { supabase } from '@/lib/supabase';
import type { User } from '@supabase/supabase-js';

const BUDGETS = [
  { label: 'Under PKR 1M',  value: 1_000_000 },
  { label: 'PKR 1M – 2M',  value: 2_000_000 },
  { label: 'PKR 2M – 3M',  value: 3_000_000 },
  { label: 'PKR 3M – 5M',  value: 5_000_000 },
  { label: 'PKR 5M+',      value: 10_000_000 },
];
const COMMUTES = [
  { label: 'Under 10 km',  value: 8 },
  { label: '10 – 30 km',   value: 20 },
  { label: '30 – 60 km',   value: 45 },
  { label: '60 km+',       value: 70 },
];
const FAMILY = [
  { label: 'Just me',      value: 1 },
  { label: '2 people',     value: 2 },
  { label: '3 – 4 people', value: 4 },
  { label: '5+ people',    value: 5 },
];
const FUELS = ['Any', 'Petrol', 'Diesel', 'Hybrid', 'CNG'];
const TRANS = ['Any', 'Automatic', 'Manual'];
const PRIORITIES = [
  { id: 'fuel_efficiency', label: '⛽ Fuel Efficiency' },
  { id: 'low_mileage',     label: '🏁 Low Mileage' },
  { id: 'latest_model',    label: '🆕 Latest Model' },
  { id: 'low_maintenance', label: '🔧 Low Maintenance' },
  { id: 'spacious',        label: '🪑 Spacious Cabin' },
  { id: 'sporty',          label: '🏎 Sporty Look' },
  { id: 'city_driving',    label: '🏙 City Friendly' },
  { id: 'reliability',     label: '🛡 Reliability' },
];

function OptionBtn({ active, disabled, onClick, children }: {
  active: boolean; disabled?: boolean; onClick: () => void; children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`flex items-center justify-between p-3.5 rounded-xl border text-sm font-medium transition-all text-left
        ${active    ? 'border-[#8b5cf6] bg-[#8b5cf6]/15 text-white'
        : disabled  ? 'border-white/5 bg-white/2 text-gray-600 opacity-40 cursor-not-allowed'
        :             'border-white/10 bg-white/5 text-gray-300 hover:border-[#8b5cf6]/40 hover:bg-white/8'}`}
    >
      <span>{children}</span>
      {active && <Check className="w-4 h-4 text-[#8b5cf6] shrink-0" />}
    </button>
  );
}

export default function ProfilePage() {
  const router = useRouter();
  const [user,     setUser]     = useState<User | null>(null);
  const [loading,  setLoading]  = useState(true);
  const [saving,   setSaving]   = useState(false);
  const [step,     setStep]     = useState(1);

  const [budgetMax,     setBudgetMax]     = useState(3_000_000);
  const [commuteKm,     setCommuteKm]     = useState(20);
  const [familySize,    setFamilySize]    = useState(2);
  const [fuel,          setFuel]          = useState('Any');
  const [transmission,  setTransmission]  = useState('Any');
  const [priorities,    setPriorities]    = useState<string[]>([]);

  useEffect(() => {
    (async () => {
      const { data: { user: u } } = await supabase.auth.getUser();
      if (!u) { router.push('/login'); return; }
      setUser(u);
      const { data } = await supabase
        .from('user_profiles').select('*').eq('user_id', u.id).single();
      if (data) {
        setBudgetMax(data.budget_max);
        setCommuteKm(data.commute_km);
        setFamilySize(data.family_size);
        setFuel(data.fuel_preference);
        setTransmission(data.transmission_preference);
        setPriorities(data.priorities || []);
      }
      setLoading(false);
    })();
  }, []);

  const togglePriority = (id: string) =>
    setPriorities(p => p.includes(id) ? p.filter(x => x !== id) : p.length < 4 ? [...p, id] : p);

  const handleSave = async () => {
    if (!user) return;
    setSaving(true);
    await supabase.from('user_profiles').upsert(
      { user_id: user.id, budget_max: budgetMax, commute_km: commuteKm,
        family_size: familySize, fuel_preference: fuel,
        transmission_preference: transmission, priorities,
        updated_at: new Date().toISOString() },
      { onConflict: 'user_id' }
    );
    setSaving(false);
    router.push('/chat');
  };

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center bg-[#050505]">
      <Loader2 className="w-10 h-10 animate-spin text-[#8b5cf6]" />
    </div>
  );

  return (
    <div className="min-h-screen bg-[#050505] text-white flex flex-col items-center justify-center p-4 relative overflow-hidden">
      <div className="absolute top-[-20%] left-[10%] w-[500px] h-[500px] bg-[#8b5cf6]/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[10%] w-[400px] h-[400px] bg-blue-500/10 rounded-full blur-[120px] pointer-events-none" />
      
      {/* Back Button */}
      <button 
        onClick={() => router.back()}
        className="absolute top-6 left-6 p-3 rounded-2xl glass-panel text-gray-400 hover:text-white hover:bg-white/10 transition-all flex items-center gap-2 group z-50"
      >
        <ChevronLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
        <span className="text-sm font-medium pr-1">Back</span>
      </button>

      <div className="w-full max-w-lg z-10">

        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-[#8b5cf6] to-blue-500 flex items-center justify-center mx-auto mb-4 shadow-lg shadow-[#8b5cf6]/30">
            <Brain className="w-7 h-7 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-white mb-1">Your Car Profile</h1>
          <p className="text-gray-400 text-sm">Help us find your perfect match in minutes.</p>
        </div>

        {/* Progress bar */}
        <div className="flex gap-2 mb-8">
          {[1,2,3,4].map(i => (
            <div key={i} className={`h-1 flex-1 rounded-full transition-all duration-500 ${i <= step ? 'bg-[#8b5cf6]' : 'bg-white/10'}`} />
          ))}
        </div>

        <div className="glass-panel rounded-3xl p-7 border border-[#8b5cf6]/20 shadow-2xl">

          {/* Step 1 – Budget */}
          {step === 1 && (
            <div className="animate-slide-up">
              <p className="text-xs font-semibold text-[#8b5cf6] uppercase tracking-widest mb-2">Step 1 of 4</p>
              <h2 className="text-2xl font-bold mb-1">What&apos;s your budget?</h2>
              <p className="text-gray-400 text-sm mb-5">Maximum you&apos;d like to spend.</p>
              <div className="grid grid-cols-1 gap-2.5">
                {BUDGETS.map(o => (
                  <OptionBtn key={o.value} active={budgetMax === o.value} onClick={() => setBudgetMax(o.value)}>
                    {o.label}
                  </OptionBtn>
                ))}
              </div>
            </div>
          )}

          {/* Step 2 – Commute & Family */}
          {step === 2 && (
            <div className="animate-slide-up">
              <p className="text-xs font-semibold text-[#8b5cf6] uppercase tracking-widest mb-2">Step 2 of 4</p>
              <h2 className="text-2xl font-bold mb-5">Your daily life</h2>
              <p className="text-sm text-gray-400 mb-3">Daily commute distance?</p>
              <div className="grid grid-cols-2 gap-2.5 mb-5">
                {COMMUTES.map(o => (
                  <OptionBtn key={o.value} active={commuteKm === o.value} onClick={() => setCommuteKm(o.value)}>{o.label}</OptionBtn>
                ))}
              </div>
              <p className="text-sm text-gray-400 mb-3">How many people ride with you?</p>
              <div className="grid grid-cols-2 gap-2.5">
                {FAMILY.map(o => (
                  <OptionBtn key={o.value} active={familySize === o.value} onClick={() => setFamilySize(o.value)}>{o.label}</OptionBtn>
                ))}
              </div>
            </div>
          )}

          {/* Step 3 – Fuel & Transmission */}
          {step === 3 && (
            <div className="animate-slide-up">
              <p className="text-xs font-semibold text-[#8b5cf6] uppercase tracking-widest mb-2">Step 3 of 4</p>
              <h2 className="text-2xl font-bold mb-5">Your preferences</h2>
              <p className="text-sm text-gray-400 mb-3">Fuel type</p>
              <div className="flex flex-wrap gap-2 mb-6">
                {FUELS.map(f => (
                  <button key={f} onClick={() => setFuel(f)}
                    className={`px-4 py-2 rounded-full border text-sm font-medium transition-all
                      ${fuel === f ? 'border-[#8b5cf6] bg-[#8b5cf6]/20 text-white' : 'border-white/10 bg-white/5 text-gray-300 hover:border-[#8b5cf6]/40'}`}>
                    {f}
                  </button>
                ))}
              </div>
              <p className="text-sm text-gray-400 mb-3">Transmission</p>
              <div className="flex gap-2.5">
                {TRANS.map(t => (
                  <button key={t} onClick={() => setTransmission(t)}
                    className={`flex-1 py-3 rounded-xl border text-sm font-medium transition-all
                      ${transmission === t ? 'border-[#8b5cf6] bg-[#8b5cf6]/20 text-white' : 'border-white/10 bg-white/5 text-gray-300 hover:border-[#8b5cf6]/40'}`}>
                    {t}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Step 4 – Priorities */}
          {step === 4 && (
            <div className="animate-slide-up">
              <p className="text-xs font-semibold text-[#8b5cf6] uppercase tracking-widest mb-2">Step 4 of 4</p>
              <h2 className="text-2xl font-bold mb-1">What matters most?</h2>
              <p className="text-gray-400 text-sm mb-5">Pick up to 4 priorities.</p>
              <div className="grid grid-cols-2 gap-2.5">
                {PRIORITIES.map(o => {
                  const on = priorities.includes(o.id);
                  return (
                    <OptionBtn key={o.id} active={on} disabled={!on && priorities.length >= 4} onClick={() => togglePriority(o.id)}>
                      {o.label}
                    </OptionBtn>
                  );
                })}
              </div>
            </div>
          )}

          {/* Nav buttons */}
          <div className="flex gap-3 mt-7">
            {step > 1 && (
              <button onClick={() => setStep(s => s - 1)}
                className="flex items-center gap-2 px-5 py-3 rounded-xl btn-secondary text-sm font-medium">
                <ChevronLeft className="w-4 h-4" /> Back
              </button>
            )}
            {step < 4 ? (
              <button onClick={() => setStep(s => s + 1)}
                className="flex-1 flex items-center justify-center gap-2 px-5 py-3 rounded-xl btn-primary text-sm font-semibold">
                Next <ChevronRight className="w-4 h-4" />
              </button>
            ) : (
              <button onClick={handleSave} disabled={saving}
                className="flex-1 flex items-center justify-center gap-2 px-5 py-3 rounded-xl btn-primary text-sm font-semibold disabled:opacity-60">
                {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <><Check className="w-4 h-4" /> Save &amp; Find My Cars</>}
              </button>
            )}
          </div>
        </div>

        <p className="text-center text-xs text-gray-600 mt-4">
          <Link href="/chat" className="hover:text-gray-400 transition-colors">← Skip for now</Link>
        </p>
      </div>
    </div>
  );
}
