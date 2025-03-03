import React, { useRef } from 'react';
import { useGSAP } from '@/hooks/useGSAP';

export const MotivationCard = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const textRef1 = useRef<HTMLParagraphElement>(null);
  const textRef2 = useRef<HTMLParagraphElement>(null);

  useGSAP((gsap) => {
    // Set initial states to completely invisible
    gsap.set([contentRef.current, textRef1.current, textRef2.current], {
      opacity: 0,
      y: 50,
      immediateRender: true
    });

    // Create scroll-triggered animation
    gsap.timeline({
      scrollTrigger: {
        trigger: containerRef.current,
        start: "top 80%",
        end: "top 20%",
        scrub: 1,
        toggleActions: "play none none reverse"
      }
    })
    .to(contentRef.current, {
      opacity: 1,
      y: 0,
      duration: 1,
      ease: "power2.out"
    })
    .to(textRef1.current, {
      opacity: 1,
      y: 0,
      duration: 1,
      ease: "power2.out"
    })
    .to(textRef2.current, {
      opacity: 1,
      y: 0,
      duration: 1,
      ease: "power2.out"
    }, "-=0.5");

  }, []);

  return (
    <div 
      ref={containerRef}
      className="absolute top-[100vh] w-full min-h-screen flex items-center justify-center bg-[#020617]"
    >
      {/* Animated Background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute inset-0 animate-gradient">
          <svg className="w-full h-full opacity-20" viewBox="0 0 100 100">
            <defs>
              <radialGradient id="gradient" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="rgb(79, 70, 229)" stopOpacity="0.2" />
                <stop offset="100%" stopColor="transparent" />
              </radialGradient>
            </defs>
            <circle cx="50" cy="50" r="45" fill="url(#gradient)" />
          </svg>
        </div>
      </div>

      {/* Content */}
      <div 
        ref={contentRef}
        className="max-w-4xl mx-auto px-4 space-y-8"
      >
        <p 
          ref={textRef1}
          className="text-xl md:text-2xl text-white leading-relaxed"
        >
          When labels are semantically very similar—such as{' '}
          <span className="text-indigo-300 font-medium">'diet'</span> versus{' '}
          <span className="text-indigo-300 font-medium">'nutrition'</span>
          —traditional LLMs may struggle, leading to overlapping embeddings and ambiguous classifications.
        </p>
        
        <p 
          ref={textRef2}
          className="text-xl md:text-2xl text-white leading-relaxed"
        >
          In this presentation, we explore both the challenges of traditional methods and a 
          novel graph-based approach that can improve both accuracy and interpretability.
        </p>
      </div>
    </div>
  );
}; 