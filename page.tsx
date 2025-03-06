// app/page.tsx
'use client';

import { motion } from 'framer-motion';
import { Rocket, Palette, Code2, Sparkles } from 'lucide-react';

export default function Home() {
  const templates = Array(6).fill(null); // Replace with actual template data

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Animated Background Elements */}
      <motion.div
        className="absolute top-20 left-20 w-72 h-72 bg-purple-500/20 rounded-full blur-3xl -z-10"
        animate={{ scale: [1, 1.2, 1] }}
        transition={{ duration: 8, repeat: Infinity }}
      />
      
      <nav className="fixed w-full p-6 flex justify-between items-center backdrop-blur-lg z-50">
        <div className="flex items-center gap-2">
          <img 
            src="/WebPersonaBlack.png" 
            alt="WebPersona" 
            className="w-10 h-10"
          />
          <span className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            TemplateHub
          </span>
        </div>
        
        <div className="hidden md:flex gap-8">
          <button className="text-slate-300 hover:text-cyan-400 transition-all">Products</button>
          <button className="text-slate-300 hover:text-cyan-400 transition-all">Showcase</button>
          <button className="text-slate-300 hover:text-cyan-400 transition-all">Pricing</button>
        </div>
        
        <button className="px-6 py-2 bg-cyan-500/20 rounded-full border border-cyan-400/50 hover:bg-cyan-500/30 transition-all">
          Get Started
        </button>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6 max-w-7xl mx-auto">
        <motion.h1 
          className="text-6xl md:text-8xl font-bold text-center mb-8 bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          Build Tomorrow's Web
        </motion.h1>
        
        <p className="text-xl text-slate-400 text-center max-w-3xl mx-auto mb-12">
          Premium Next.js templates crafted with perfection. Accelerate your development process with futuristic designs.
        </p>

        <div className="flex justify-center gap-4">
          <motion.button 
            className="px-8 py-4 bg-cyan-500 rounded-lg font-semibold flex items-center gap-2 hover:bg-cyan-400 transition-all"
            whileHover={{ scale: 1.05 }}
          >
            <Rocket size={20} />
            Explore Templates
          </motion.button>
        </div>
      </section>

      {/* Templates Grid */}
      <section className="px-6 max-w-7xl mx-auto py-20">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {templates.map((_, index) => (
            <motion.div 
              key={index}
              className="group relative bg-slate-800/50 rounded-xl p-6 backdrop-blur-lg border border-slate-700/50 hover:border-cyan-500/30 transition-all"
              whileHover={{ y: -10 }}
            >
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-purple-500/10 rounded-xl opacity-0 group-hover:opacity-100 transition-all" />
              <div className="h-64 bg-slate-700/50 rounded-lg mb-4"></div>
              <h3 className="text-xl font-semibold text-slate-200 mb-2">Template {index + 1}</h3>
              <p className="text-slate-400 mb-4">Modern and clean design with advanced features</p>
              <div className="flex justify-between items-center">
                <span className="text-cyan-400 font-semibold">$149</span>
                <button className="px-4 py-2 bg-cyan-500/20 rounded-lg border border-cyan-500/30 hover:bg-cyan-500/30 transition-all">
                  Preview
                </button>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-6 bg-slate-900/50">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl font-bold text-center mb-16 text-slate-200">Why Choose Us?</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <motion.div 
              className="p-8 rounded-xl bg-slate-800/50 border border-slate-700/50 hover:border-cyan-500/30 transition-all"
              whileHover={{ scale: 1.02 }}
            >
              <Palette className="w-12 h-12 text-cyan-400 mb-6" />
              <h3 className="text-2xl font-semibold text-slate-200 mb-4">Modern Aesthetic</h3>
              <p className="text-slate-400">Cutting-edge designs following the latest web trends</p>
            </motion.div>
            
            <motion.div 
              className="p-8 rounded-xl bg-slate-800/50 border border-slate-700/50 hover:border-cyan-500/30 transition-all"
              whileHover={{ scale: 1.02 }}
            >
              <Code2 className="w-12 h-12 text-cyan-400 mb-6" />
              <h3 className="text-2xl font-semibold text-slate-200 mb-4">Clean Code</h3>
              <p className="text-slate-400">Well-structured and maintainable codebase</p>
            </motion.div>

            <motion.div 
              className="p-8 rounded-xl bg-slate-800/50 border border-slate-700/50 hover:border-cyan-500/30 transition-all"
              whileHover={{ scale: 1.02 }}
            >
              <Sparkles className="w-12 h-12 text-cyan-400 mb-6" />
              <h3 className="text-2xl font-semibold text-slate-200 mb-4">Premium Features</h3>
              <p className="text-slate-400">Advanced functionality out of the box</p>
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  );
}