"use client";

import { useState } from 'react';
import { verifyProduct } from '@/services/api';

export default function UploadBox({ 
    onResult 
}: { 
    onResult: (res: any) => void 
}) {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [brand, setBrand] = useState<string>('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [loadingStatus, setLoadingStatus] = useState<string>('Analyzing product...');

    const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selected = e.target.files?.[0];
        if (selected) {
            setFile(selected);
            setPreview(URL.createObjectURL(selected));
            setError(null);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        const dropped = e.dataTransfer.files?.[0];
        if (dropped) {
            setFile(dropped);
            setPreview(URL.createObjectURL(dropped));
            setError(null);
        }
    };

    const submit = async () => {
        if (!file) {
            setError('Please upload a product image first.');
            return;
        }
        if (!brand) {
            setError('Please select a brand.');
            return;
        }

        setIsLoading(true);
        setError(null);
        setLoadingStatus('Analyzing product...');
        
        try {
            const apiResult = await verifyProduct(file, brand);
            onResult(apiResult);
        } catch (error) {
            console.error("Verification failed", error);
            const errorMessage = error instanceof Error ? error.message : "Verification failed";
            
            if (errorMessage.includes("timeout")) {
                setError(`${errorMessage} Please try again with a clearer product image.`);
            } else if (errorMessage.includes("not yet supported")) {
                setError(`This brand is currently not supported. Please check the available brands.`);
            } else if (errorMessage.includes("not an image")) {
                setError(`Invalid file format. Please upload a valid image file (PNG, JPG, WEBP).`);
            } else if (errorMessage.includes("backend")) {
                setError(`Backend service is not running. Please ensure the verification server is online.`);
            } else {
                setError(`Analysis failed: ${errorMessage}`);
            }
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="upload-card" style={{ position: 'relative' }}>
            {isLoading && (
                <div className="loading-overlay show">
                    <div className="spinner"></div>
                    <div className="loading-text">{loadingStatus}</div>
                    <div style={{ fontSize: '0.9rem', marginTop: '0.5rem', opacity: 0.8 }}>
                        This typically takes 3-5 seconds
                    </div>
                </div>
            )}

            {error && (
                <div style={{
                    backgroundColor: '#fee',
                    border: '1px solid #faa',
                    borderRadius: '4px',
                    padding: '12px',
                    marginBottom: '16px',
                    color: '#c33',
                    fontSize: '0.9rem'
                }}>
                    <strong>⚠️ Error:</strong> {error}
                    {error.includes("timeout") && (
                        <div style={{ marginTop: '8px' }}>
                            <button 
                                onClick={submit} 
                                style={{
                                    background: '#c33',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '4px',
                                    padding: '6px 12px',
                                    cursor: 'pointer'
                                }}
                                disabled={isLoading}
                            >
                                Try Again
                            </button>
                        </div>
                    )}
                </div>
            )}

            <div 
                className="upload-zone" 
                onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add('drag-over') }}
                onDragLeave={(e) => { e.preventDefault(); e.currentTarget.classList.remove('drag-over') }}
                onDrop={handleDrop}
            >
                <input type="file" accept="image/*" onChange={handleFile} />
                <span className="upload-icon">🖼️</span>
                <h3>Drop your product image here</h3>
                <p>or <span className="browse">browse files</span> — PNG, JPG, WEBP up to 20MB</p>
            </div>

            {preview && (
                <div className="img-preview show" style={{ position: 'relative', overflow: 'hidden' }}>
                    <div className="img-preview-label">Preview</div>
                    <img src={preview} alt="Preview" />
                    {isLoading && <div className="scan-line"></div>}
                </div>
            )}

            <div className="brand-selector">
                <label>Select Brand</label>
                <select value={brand} onChange={(e) => setBrand(e.target.value)}>
                    <option value="">— Choose a brand —</option>
                    <option value="Nike">Nike</option>
                    <option value="Adidas">Adidas</option>
                    <option value="Gucci">Gucci</option>
                    <option value="Hoka">Hoka</option>
                    <option value="Timberland">Timberland</option>
                    <option value="Bottega Veneta">Bottega Veneta</option>
                    <option value="Celine">Celine</option>
                    <option value="Vans">Vans</option>
                    <option value="Versace">Versace</option>
                    <option value="Valentino">Valentino</option>
                    <option value="Maison Margiela">Maison Margiela</option>
                    <option value="Converse">Converse</option>
                    <option value="Rick Owens">Rick Owens</option>
                    <option value="New Balance">New Balance</option>
                    <option value="Salomon">Salomon</option>
                    <option value="Louis Vuitton">Louis Vuitton</option>
                    <option value="Puma">Puma</option>
                    <option value="Asics">Asics</option>
                    <option value="Yeezy">Yeezy</option>
                    <option value="Fendi">Fendi</option>
                    <option value="Prada">Prada</option>
                </select>
            </div>

            <button 
                className="verify-btn" 
                onClick={() => {
                    void submit();
                }}
                disabled={isLoading}
            >
                🔍 Verify Authenticity
            </button>
        </div>
    );
}
