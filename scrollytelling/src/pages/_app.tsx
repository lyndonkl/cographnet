import '@/styles/globals.css';
import type { AppProps } from 'next/app';
import { useEffect } from 'react';
import { Inter } from 'next/font/google';

const inter = Inter({
  subsets: ['latin'],
  weight: ['300', '400', '500', '600', '700'],
  display: 'swap',
});

export default function App({ Component, pageProps }: AppProps) {
  useEffect(() => {
    // Ensure dark mode is applied
    document.documentElement.classList.add('dark');
    
    const initGSAP = async () => {
      const { gsap } = await import('gsap');
      const { ScrollTrigger } = await import('gsap/ScrollTrigger');
      gsap.registerPlugin(ScrollTrigger);
    };
    
    initGSAP();
  }, []);

  return (
    <main className={`${inter.className} dark bg-[#020617] text-slate-50`}>
      <Component {...pageProps} />
    </main>
  );
} 