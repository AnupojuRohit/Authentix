"use client";

import { useEffect, useState } from "react";

export default function ResultPanel({ result }: { result: any }) {
    const [score, setScore] = useState(0);

    useEffect(() => {
        if (result && result.confidence) {
            let s = 0;
            const target = result.confidence;
            const step = target / 60;
            const t = setInterval(() => {
                s += step;
                if (s >= target) {
                    s = target;
                    clearInterval(t);
                }
                setScore(Math.floor(s));
            }, 20);
            return () => clearInterval(t);
        }
    }, [result]);

    if (!result) {
        return (
            <div className="result-panel">
                <h3>VERIFICATION RESULT</h3>
                <div className="result-placeholder">
                    <span>🛡️</span>
                    Upload an image and select a brand to see your authenticity report.
                </div>
            </div>
        );
    }

    const { prediction, confidence, brand_queried } = result;
    const isAuthentic = prediction === "Authentic";
    
    // Circle SVG Math
    const circumference = 326.7;
    const offset = circumference - (confidence / 100) * circumference;

    return (
        <div className={`result-panel ${!isAuthentic ? 'fake-shake' : ''}`}>
            <h3>VERIFICATION RESULT</h3>
            <div className="result-box show">
                <div className="authenticity-score">
                    <div className="score-ring">
                        <svg viewBox="0 0 120 120" width="120" height="120">
                            <circle cx="60" cy="60" r="52" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="8" />
                            <circle
                                cx="60" cy="60" r="52" fill="none"
                                stroke={isAuthentic ? "var(--accent)" : "var(--accent3)"}
                                strokeWidth="8" strokeLinecap="round"
                                strokeDasharray={circumference}
                                strokeDashoffset={offset}
                                style={{ transition: "stroke-dashoffset 1.5s ease" }}
                            />
                        </svg>
                        <div className="score-val">
                            {score}<small>%</small>
                        </div>
                    </div>
                    
                    <div>
                        {isAuthentic ? (
                            <span className="verdict authentic">✓ Authentic Product</span>
                        ) : (
                            <span className="verdict fake">✗ Likely Counterfeit</span>
                        )}
                    </div>
                    
                    <div style={{ fontSize: '0.75rem', color: 'var(--muted)', marginTop: '0.5rem' }}>
                        Compared against database.
                    </div>
                </div>

                <div className="result-details">
                    <div className="detail-row">
                        <span className="detail-label">Logo Match</span>
                        <span className={`detail-value ${isAuthentic ? 'tag-good' : 'tag-bad'}`}>{confidence}%</span>
                    </div>
                    <div className="detail-row">
                        <span className="detail-label">Quality Analysis</span>
                        <span className={`detail-value ${isAuthentic ? 'tag-good' : 'tag-bad'}`}>
                            {isAuthentic ? 'Verified' : 'Mismatch'}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
}
