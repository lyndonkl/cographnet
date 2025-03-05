/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Background & Surfaces
        background: 'var(--color-background)',
        card: 'var(--color-card)',
        
        // Text colors
        'text-body': 'var(--color-text-body)',
        'text-heading': 'var(--color-text-heading)',
        
        // Accents & Highlights
        'accent-primary': 'var(--color-accent-primary)',
        'accent-secondary': 'var(--color-accent-secondary)',
        'accent-support': 'var(--color-accent-support)',
      },
      spacing: {
        'card-y': '3rem', // 48px vertical padding
        'card-x': '2rem', // 32px horizontal padding
      },
      fontSize: {
        'body': ['1.125rem', '1.75'], // 18px with 1.75 line height
        'h1': ['2.5rem', '1.2'],      // 40px
        'h2': ['2rem', '1.2'],        // 32px
        'h3': ['1.75rem', '1.2'],     // 28px
      },
      boxShadow: {
        'card': '0 0 20px rgba(126,224,206,0.2)', // Pastel teal glow
      },
      borderRadius: {
        'card': '0.5rem', // 8px rounded corners
      },
      transitionDuration: {
        'default': '200ms',
      },
    },
  },
  plugins: [],
} 