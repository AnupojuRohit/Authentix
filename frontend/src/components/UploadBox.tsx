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

    const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selected = e.target.files?.[0];
        if (selected) {
            setFile(selected);
            setPreview(URL.createObjectURL(selected));
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        const dropped = e.dataTransfer.files?.[0];
        if (dropped) {
            setFile(dropped);
            setPreview(URL.createObjectURL(dropped));
        }
    };

    const submit = async () => {
        if (!file) {
            alert('Please upload a product image first.');
            return;
        }
        if (!brand) {
            alert('Please select a brand.');
            return;
        }

        setIsLoading(true);
        try {
            const apiResult = await verifyProduct(file, brand);
            onResult(apiResult);
        } catch (error) {
            console.error("Verification failed", error);
            alert("Verification failed. Make sure the backend is running.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="upload-card" style={{ position: 'relative' }}>
            {isLoading && (
                <div className="loading-overlay show">
                    <div className="spinner"></div>
                    <div className="loading-text">Analyzing product...</div>
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
                    <option value="Louis Vuitton">Louis Vuitton</option>
                    <option value="Supreme">Supreme</option>
                    <option value="Puma">Puma</option>
                    <option value="Balenciaga">Balenciaga</option>
                    <option value="Off-White">Off-White</option>
                    <option value="New Balance">New Balance</option>
                    <option value="Versace">Versace</option>
                    <option value="Ralph Lauren">Ralph Lauren</option>
                    <option value="Stone Island">Stone Island</option>
                </select>
            </div>

            <button 
                className="verify-btn" 
                onClick={submit}
                disabled={isLoading}
            >
                🔍 Verify Authenticity
            </button>
        </div>
    );
}
