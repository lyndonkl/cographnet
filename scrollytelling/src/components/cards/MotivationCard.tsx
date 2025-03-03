import React from 'react';
import { BaseCard } from './BaseCard';

export const MotivationCard = () => {
  return (
    <BaseCard>
      <p className="text-body">
        When labels are semantically very similar—such as{' '}
        <span className="highlight">'diet'</span> versus{' '}
        <span className="highlight">'nutrition'</span>
        —traditional LLMs may struggle, leading to overlapping embeddings and ambiguous classifications.
      </p>
      
      <p className="text-body">
        In this presentation, we explore both the challenges of traditional methods and a 
        novel graph-based approach that can improve both accuracy and interpretability.
      </p>
    </BaseCard>
  );
}; 