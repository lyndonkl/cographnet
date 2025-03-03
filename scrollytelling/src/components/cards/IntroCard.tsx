import React, { useRef } from 'react';
import { useGSAP } from '@/hooks/useGSAP';

export const IntroCard = () => {
  const titleRef = useRef<HTMLHeadingElement>(null);

  useGSAP((gsap) => {
    gsap.set(titleRef.current, {
      opacity: 0,
      y: 50
    });

    gsap.to(titleRef.current, {
      opacity: 1,
      y: 0,
      duration: 2,
      ease: "power2.out"
    });

    // Add scroll animation for title
    gsap.to(titleRef.current, {
      scrollTrigger: {
        trigger: "body",
        start: "top top",
        end: "+=100",
        scrub: true
      },
      scale: 0.8,
      y: "20px",
    });
  }, []);

  return (
    <div className="fixed top-0 left-0 w-full z-50 pt-6 bg-gradient-to-b from-[#020617] via-[#020617] to-transparent">
      <h1 
        ref={titleRef}
        className="text-3xl md:text-5xl lg:text-6xl font-bold text-white text-center max-w-5xl mx-auto px-4 transition-all duration-300"
      >
        Overcoming Classification Challenges with Graph Representations
      </h1>
    </div>
  );
}; 