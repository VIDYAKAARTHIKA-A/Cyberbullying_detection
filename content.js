setInterval(() => {
    const comments = Array.from(document.querySelectorAll("p"))
        .map(el => el.innerText)
        .filter(text => text.length > 5);

    comments.forEach(async (text) => {
        const response = await fetch("http://localhost:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        const result = await response.json();
        if (result.label === "bullying") {
            console.warn("⚠️ Bullying Detected:", text);
            // optionally highlight text or alert user
        }
    });
}, 5000);
