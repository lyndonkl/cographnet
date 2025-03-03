import { MainLayout } from '@/components/layout/MainLayout';
import { CardGroup } from '@/components/cards/CardGroup';
import { EmbeddingScatter } from '@/components/visualizations/EmbeddingScatter';

export default function Home() {
  return (
    <MainLayout>
      {/* First Group */}
      <CardGroup 
        title="Overcoming Classification Challenges with Graph Representations"
        isFirst={true}
        content={
          <div className="space-y-8">
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
          </div>
        }
      />

      {/* Second Group */}
      <CardGroup 
        title="Challenges with Traditional LLMs"
        content={
          <div className="space-y-12">
            <p className="text-body">
              Transformer-based models excel at many tasks, yet they can struggle when classes are semantically close. 
              For example, when classifying text into <span className="highlight">'diet'</span> versus{' '}
              <span className="highlight">'nutrition'</span>, the embeddings may overlap, causing misclassifications.
            </p>
            <EmbeddingScatter />
          </div>
        }
      />
    </MainLayout>
  );
} 