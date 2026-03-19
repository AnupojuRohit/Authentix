"use client";

import { useEffect, useState } from "react";

// ── Type matching the flattened Step 6 backend response ──────────────────────
interface ApiResponse {
    verdict: "authentic" | "fake";
    confidence: number;
    authentic_probability: number;
    fake_probability: number;
    confidence_level: "high" | "medium" | "low";
    heatmap_base64: string;
    analysis_regions: string[];
    processing_time_ms: number;
    layer_scores?: Record<string, number>;
}

// ── Helpers ─────────────────────────────────────────────────────────────────
function verdictLabel(verdict: string) {
    if (verdict === "authentic") return { text: "✓ Authentic Product", cls: "authentic" };
    return { text: "✗ Likely Counterfeit", cls: "fake" };
}

function getConfidenceColor(level: string) {
    if (level === "high") return "var(--tag-good-bg, #059669)";
    if (level === "medium") return "var(--tag-warn-bg, #d97706)";
    return "var(--tag-bad-bg, #dc2626)";
}

// ── Component ────────────────────────────────────────────────────────────────
export default function ResultPanel({ result }: { result: ApiResponse | null }) {
    const [animatedScore, setAnimatedScore] = useState(0);

    useEffect(() => {
        if (!result) return;
        const target = Math.max(0, Math.min(100, result.confidence ?? 0));
        let s = 0;
        const step = target / 60;
        const t = setInterval(() => {
            s += step;
            if (s >= target) { s = target; clearInterval(t); }
            setAnimatedScore(Math.floor(s));
        }, 20);
        return () => clearInterval(t);
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

    const { verdict, confidence, confidence_level, heatmap_base64, analysis_regions, processing_time_ms } = result;
    const isAuthentic = verdict === "authentic";
    const isInconclusive = confidence < 60;

    const circumference = 326.7;
    const safeScore = isFinite(confidence) ? Math.max(0, Math.min(100, confidence)) : 0;
    const offset = circumference - (safeScore / 100) * circumference;
    const { text: verdictText, cls: verdictCls } = verdictLabel(verdict);

    return (
        <div className={`result-panel ${!isAuthentic ? "fake-shake" : ""}`}>
            <h3>VERIFICATION RESULT</h3>
            <div className="result-box show">

                {/* ── Score Ring & Verdict ────────────────────────────────── */}
                <div className="authenticity-score">
                    <div className="score-ring">
                        <svg viewBox="0 0 120 120" width="120" height="120">
                            <circle cx="60" cy="60" r="52" fill="none"
                                stroke="rgba(255,255,255,0.05)" strokeWidth="8" />
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
                            {animatedScore}<small>%</small>
                        </div>
                    </div>

                    <div style={{ textAlign: "center" }}>
                        <span className={`verdict ${verdictCls}`} style={{ display: "block" }}>
                            {isInconclusive ? "Inconclusive" : verdictText}
                        </span>
                        <span style={{ 
                            fontSize: "0.7rem", 
                            textTransform: "uppercase", 
                            background: getConfidenceColor(confidence_level),
                            padding: "2px 8px",
                            borderRadius: "12px",
                            color: "white",
                            fontWeight: 700,
                            marginTop: "6px",
                            display: "inline-block"
                        }}>
                            {confidence_level} Confidence
                        </span>
                    </div>

                    {isInconclusive && (
                        <div style={{ fontSize: "0.75rem", color: "var(--accent3)", marginTop: "0.8rem", textAlign: "center", lineHeight: "1.4" }}>
                            ⚠️ Analysis quality is too low for a definitive verdict. Please try a clearer product image.
                        </div>
                    )}

                    <div style={{ fontSize: "0.75rem", color: "var(--muted)", marginTop: "1rem" }}>
                        Analysis complete in {processing_time_ms}ms
                    </div>
                </div>

                {/* ── Analysis Regions ────────────────────────────────────── */}
                <div className="result-details">
                    <p style={{ fontSize: "0.75rem", color: "var(--muted)", marginBottom: "0.6rem" }}>Regions Analyzed</p>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                        {analysis_regions.map((region, i) => (
                            <span key={i} style={{ 
                                fontSize: "0.7rem", 
                                background: "rgba(255,255,255,0.05)", 
                                padding: "4px 8px", 
                                borderRadius: "4px",
                                border: "1px solid rgba(255,255,255,0.1)"
                            }}>
                                {region}
                            </span>
                        ))}
                    </div>
                </div>

                {/* ── Real AI Heatmap ─────────────────────────────────────── */}
                {heatmap_base64 && (
                    <div style={{ marginTop: "1.5rem" }}>
                        <p style={{ fontSize: "0.75rem", color: "var(--muted)", marginBottom: "0.6rem" }}>
                            AI Focus Map (Grad-CAM)
                        </p>
                        <div style={{ position: "relative", borderRadius: "8px", overflow: "hidden", border: "1px solid rgba(255,255,255,0.1)" }}>
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img
                                src={heatmap_base64}
                                alt="AI Analysis Heatmap"
                                style={{ width: "100%", display: "block" }}
                            />
                            <div style={{ 
                                position: "absolute", 
                                bottom: "8px", 
                                right: "8px", 
                                fontSize: "0.6rem", 
                                background: "rgba(0,0,0,0.6)", 
                                padding: "2px 6px", 
                                borderRadius: "4px"
                            }}>
                                High Weight Focus
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
