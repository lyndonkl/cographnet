export default function App() {
  return (
    <div style={{ minHeight: '100vh', padding: '2rem' }}>
      {/* Main Content Area */}
      <div className="content-section" style={{ maxWidth: '56rem', margin: '0 auto' }}>
        
        {/* Headers Demonstration */}
        <section className="card text-section">
          <h1>Main Heading h1</h1>
          <h2>Secondary Heading h2</h2>
          <h3>Tertiary Heading h3</h3>
          <p style={{ color: 'var(--color-accent-secondary)' }}>Subheading Text Example</p>
        </section>

        {/* Card Example */}
        <section className="card text-section">
          <h2>Card Title</h2>
          <p>
            This is body text in off-white (#EAEAEA) with proper line height.
            It should be highly readable against the dark card background.
          </p>
          <p>
            Here's a <span className="highlight">highlighted piece of text</span> using
            our pastel mint accent.
          </p>
        </section>

        {/* Interactive Elements */}
        <section className="card text-section">
          <h2>Interactive Elements</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <button className="btn-primary">
              Primary Action Button
            </button>
            <div>
              <a href="#" className="link">
                Interactive Link Example
              </a>
            </div>
          </div>
        </section>

        {/* Color Palette */}
        <section className="card text-section">
          <h2>Color Palette</h2>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
            gap: '1rem', 
            marginTop: '1rem' 
          }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ height: '5rem', backgroundColor: 'var(--color-background)', borderRadius: '0.5rem' }}></div>
              <p>Background (#1F1F1F)</p>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ height: '5rem', backgroundColor: 'var(--color-card)', borderRadius: '0.5rem' }}></div>
              <p>Card (#2A2A2A)</p>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ height: '5rem', backgroundColor: 'var(--color-text-heading)', borderRadius: '0.5rem' }}></div>
              <p>Heading (#7EE0CE)</p>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ height: '5rem', backgroundColor: 'var(--color-accent-primary)', borderRadius: '0.5rem' }}></div>
              <p>Primary (#F8B4D9)</p>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ height: '5rem', backgroundColor: 'var(--color-accent-secondary)', borderRadius: '0.5rem' }}></div>
              <p>Secondary (#C3A4F3)</p>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ height: '5rem', backgroundColor: 'var(--color-accent-support)', borderRadius: '0.5rem' }}></div>
              <p>Support (#B5E3C6)</p>
            </div>
          </div>
        </section>

        {/* Typography */}
        <section className="card text-section">
          <h2>Typography</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <h1>Heading 1</h1>
            <h2>Heading 2</h2>
            <h3>Heading 3</h3>
            <p>Body Text</p>
          </div>
        </section>
      </div>
    </div>
  )
}
