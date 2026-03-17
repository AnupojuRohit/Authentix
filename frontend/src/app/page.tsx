"use client";

import { useState } from "react";
import UploadBox from "@/components/UploadBox";
import ResultPanel from "@/components/ResultPanel";

export default function Home() {
  const [result, setResult] = useState<any>(null);

  return (
    <main>
      {/* Custom Cursor (CSS handled) */}
      <div className="cursor" id="cursor"></div>
      <div className="cursor-ring" id="cursorRing"></div>

      {/* Nav */}
      <nav>
        <div className="logo">
          Auth<span>en</span>tix
        </div>
        <div className="nav-links">
          <a href="#how">How It Works</a>
          <a href="#verify">Verify Now</a>
          <a href="#brands">Brands</a>
          <a href="#features">Features</a>
        </div>
        <button
          className="nav-cta"
          onClick={() => document.getElementById("verify")?.scrollIntoView({ behavior: "smooth" })}
        >
          Try Free
        </button>
      </nav>

      {/* Hero */}
      <section className="hero">
        <div className="hero-bg">
          <div className="bg-image"></div>
          <div className="overlay-gradient"></div>
        </div>
        <div className="hero-grid"></div>
        <div className="hero-content">
          <div className="hero-badge">AI-Powered Brand Verification</div>
          <h1>
            <span className="line1">IS YOUR</span>
            <span className="line2">PRODUCT</span>
            <span className="line3 glitch" data-text="REAL?">
              REAL?
            </span>
          </h1>
          <p className="hero-sub">
            Upload any product image and let Authentix verify it against our live brand database. Stop fakes before they reach you.
          </p>
          <div className="hero-actions">
            <button
              className="btn-primary"
              onClick={() => document.getElementById("verify")?.scrollIntoView({ behavior: "smooth" })}
            >
              Verify a Product →
            </button>
          </div>
        </div>
      </section>

      {/* Stats Bar */}
      <div className="stats-bar">
        <div className="stat">
          <div className="stat-num">2.4M</div>
          <div className="stat-label">Products Scanned</div>
        </div>
        <div className="stat">
          <div className="stat-num">340</div>
          <div className="stat-label">Brands Protected</div>
        </div>
        <div className="stat">
          <div className="stat-num">99.2</div>
          <div className="stat-label">Accuracy %</div>
        </div>
      </div>

      {/* Upload / Verify */}
      <section className="upload-section" id="verify">
        <div className="container">
          <div className="section-label">// Verify Now</div>
          <h2 className="section-title">
            CHECK YOUR<br />PRODUCT
          </h2>
          <div className="upload-wrapper">
            <UploadBox onResult={(data) => setResult(data)} />
            <ResultPanel result={result} />
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer>
        <div className="footer-logo">AUTHENTIX</div>
        <div className="footer-text">© 2026 Authentix. Real brands deserve real protection.</div>
      </footer>
    </main>
  );
}
