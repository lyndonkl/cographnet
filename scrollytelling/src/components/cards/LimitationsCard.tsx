import React from 'react';
import { BaseCard } from './BaseCard';
import { EmbeddingScatter } from '../visualizations/EmbeddingScatter';

export const LimitationsCard = () => {
  return (
    <div className="relative min-h-[200vh]">
      <div className="fixed top-0 left-0 w-full z-50 pt-6 bg-gradient-to-b from-[#020617] via-[#020617] to-transparent">
        <h2 className="title-large">
          Challenges with Traditional LLMs
        </h2>
      </div>

      <BaseCard startPosition="top-[100vh]">
        <div className="space-y-12">
          <p className="text-body">
            Transformer-based models excel at many tasks, yet they can struggle when classes are semantically close. 
            For example, when classifying text into <span className="highlight">'diet'</span> versus{' '}
            <span className="highlight">'nutrition'</span>, the embeddings may overlap, causing misclassifications.
          </p>

          <EmbeddingScatter />
        </div>
      </BaseCard>
    </div>
  );
}; 