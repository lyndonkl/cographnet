import Document, { Html, Head, Main, NextScript, DocumentContext } from 'next/document'

class MyDocument extends Document {
  static async getInitialProps(ctx: DocumentContext) {
    const initialProps = await Document.getInitialProps(ctx)
    return { ...initialProps }
  }

  render() {
    return (
      <Html lang="en" className="dark">
        <Head>
          <script
            dangerouslySetInnerHTML={{
              __html: `
                try {
                  document.documentElement.classList.add('dark');
                  document.documentElement.style.backgroundColor = '#020617';
                  document.documentElement.style.color = '#f8fafc';
                } catch (_) {}
              `,
            }}
          />
        </Head>
        <body className="bg-[#020617] text-slate-50">
          <Main />
          <NextScript />
        </body>
      </Html>
    )
  }
}

export default MyDocument 