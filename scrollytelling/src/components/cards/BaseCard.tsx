import React, { useRef } from 'react';
import { useGSAP } from '@/hooks/useGSAP';

interface BaseCardProps {
  children: React.ReactNode;
  className?: string;
  startPosition?: string;
  scrollTrigger?: {
    start?: string;
    end?: string;
    scrub?: boolean | number;
    toggleActions?: string;
  };
}

export const BaseCard = ({ 
  children, 
  className = '', 
  startPosition = 'top-[100vh]',
  scrollTrigger = {
    start: "top 80%",
    end: "top 20%",
    scrub: 1,
    toggleActions: "play none none reverse"
  }
}: BaseCardProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  useGSAP((gsap) => {
    gsap.set(contentRef.current, {
      opacity: 0,
      y: 50,
      immediateRender: true
    });

    gsap.timeline({
      scrollTrigger: {
        trigger: containerRef.current,
        ...scrollTrigger
      }
    })
    .to(contentRef.current, {
      opacity: 1,
      y: 0,
      duration: 1,
      ease: "power2.out"
    });
  }, []);

  return (
    <div 
      ref={containerRef}
      className={`card-container ${startPosition} ${className}`}
    >
      <div ref={contentRef} className="card-content">
        {children}
      </div>
    </div>
  );
}; 