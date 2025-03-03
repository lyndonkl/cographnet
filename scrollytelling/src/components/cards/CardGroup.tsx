import React, { useRef } from 'react';
import { useGSAP } from '@/hooks/useGSAP';
import { BaseCard } from './BaseCard';

interface CardGroupProps {
  title: string;
  content: React.ReactNode;
  startPosition?: string;
  isFirst?: boolean;
}

export const CardGroup = ({ 
  title, 
  content, 
  startPosition = "top-[100vh]",
  isFirst = false 
}: CardGroupProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const titleRef = useRef<HTMLHeadingElement>(null);

  useGSAP((gsap) => {
    // Initial state
    gsap.set(titleRef.current, {
      opacity: 0,
      y: 50
    });

    // Title animation timeline
    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: containerRef.current,
        start: "top center",
        end: "bottom center",
        toggleActions: "play reverse play reverse",
        scrub: 1,
      }
    });

    // Fade in and transform title
    tl.to(titleRef.current, {
      opacity: 1,
      y: 0,
      duration: 1,
      ease: "power2.out"
    });

    // Fade out title when scrolling past this section
    gsap.to(titleRef.current, {
      scrollTrigger: {
        trigger: containerRef.current,
        start: "bottom center",
        end: "bottom top",
        scrub: 1,
      },
      opacity: 0,
      y: -50,
    });
  }, []);

  return (
    <div ref={containerRef} className="relative min-h-[200vh]">
      {/* Fixed Title */}
      <div className="fixed top-0 left-0 w-full z-50 pt-16 md:pt-20 bg-gradient-to-b from-[#020617] via-[#020617] to-transparent">
        <h1 ref={titleRef} className="title-large">
          {title}
        </h1>
      </div>

      {/* Scrolling Content */}
      <BaseCard startPosition={startPosition}>
        {content}
      </BaseCard>
    </div>
  );
}; 