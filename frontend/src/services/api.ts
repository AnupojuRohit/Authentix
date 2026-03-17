export const verifyProduct = async (imageFile: File, brand: string) => {
    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("brand", brand);

    // Using the 8000 port created in Step 1
    const response = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        throw new Error("API request failed");
    }

    return await response.json();
};
