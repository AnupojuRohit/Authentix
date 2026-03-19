import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Authentix — Real or Fake?',
  description: 'AI-Powered Brand Verification: Let Authentix verify it against our live brand database.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  )
}
