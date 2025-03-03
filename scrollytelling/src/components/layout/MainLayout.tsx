import React, { ReactNode } from 'react';

interface MainLayoutProps {
  children: ReactNode;
}

export const MainLayout = ({ children }: MainLayoutProps) => {
  return (
    <main className="min-h-screen w-full bg-[#020617] text-white">
      {children}
    </main>
  );
}; 