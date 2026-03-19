const API_BASE_URL = (
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");
const REQUEST_TIMEOUT = 60000; // 60 seconds timeout
const MAX_RETRIES = 2;

// Helper to abort long-running requests cleanly.
async function fetchWithTimeout(url: string, options: RequestInit = {}, timeout = REQUEST_TIMEOUT): Promise<Response> {
    const controller = new AbortController();
    const timerId = setTimeout(() => controller.abort(), timeout);

    try {
        return await fetch(url, {
            ...options,
            mode: "cors",
            signal: controller.signal,
            cache: "no-store",
        });
    } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
            throw new Error("Request timeout");
        }
        throw error;
    } finally {
        clearTimeout(timerId);
    }
}

// Retry logic for transient failures
async function fetchWithRetry(url: string, options: RequestInit = {}, retries = MAX_RETRIES): Promise<Response> {
    for (let i = 0; i <= retries; i++) {
        try {
            const response = await fetchWithTimeout(url, options);
            if (response.ok || i === retries) {
                return response;
            }
            // Only retry transient gateway/overload errors (502/503/504).
            // 500 Internal Server Errors from ML inference are deterministic — retrying wastes time.
            if (response.status === 502 || response.status === 503 || response.status === 504) {
                console.warn(`Transient server error (${response.status}), retrying... (${i + 1}/${retries})`);
                await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1))); // Exponential backoff
                continue;
            }
            // For all other non-ok statuses (400, 422, 500, etc.) return immediately — no retry.
            return response;
        } catch (error) {
            if (i === retries) throw error;
            console.warn(`Request failed, retrying... (${i + 1}/${retries})`, error);
            await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1))); // Exponential backoff
        }
    }
    throw new Error("Max retries exceeded");
}

export const verifyProduct = async (imageFile: File, brand: string) => {
    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("brand", brand);

    try {
        const response = await fetchWithRetry(`${API_BASE_URL}/predict/`, {
            method: "POST",
            body: formData,
            headers: {
                Accept: "application/json",
            },
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            // Three-tiered fallback: FastAPI `detail` → generic `message` → human-readable default
            const msg =
                errorData?.detail ||
                errorData?.message ||
                `Verification failed (server error ${response.status}). Please try again.`;
            throw new Error(msg);
        }

        return await response.json();
    } catch (error) {
        if (error instanceof TypeError && error.message.toLowerCase().includes("failed to fetch")) {
            throw new Error(`Unable to reach backend API at ${API_BASE_URL}. Ensure backend server is running and CORS is enabled.`);
        }
        if (error instanceof Error && error.message === "Request timeout") {
            throw new Error("Request timed out. The analysis is taking longer than expected. Please try again.");
        }
        throw error;
    }
};
