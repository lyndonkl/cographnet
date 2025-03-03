import { MainLayout } from '@/components/layout/MainLayout';
import { IntroCard } from '@/components/cards/IntroCard';
import { MotivationCard } from '@/components/cards/MotivationCard';

export default function Home() {
  return (
    <MainLayout>
      <div className="relative min-h-[200vh]"> {/* Force scrollable height */}
        <IntroCard />
        <MotivationCard />
      </div>
    </MainLayout>
  );
} 