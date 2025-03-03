import { useEffect } from 'react';
import type { GSAPTimeline } from 'gsap';

type AnimationCallback = (gsap: any) => void;

export const useGSAP = (animation: AnimationCallback, dependencies: any[] = []) => {
  useEffect(() => {
    let ctx: any;
    
    const initAnimation = async () => {
      const { gsap } = await import('gsap');
      const { ScrollTrigger } = await import('gsap/ScrollTrigger');
      gsap.registerPlugin(ScrollTrigger);
      ctx = gsap.context(() => animation(gsap));
    };

    initAnimation();
    
    return () => ctx?.revert();
  }, dependencies);
}; 