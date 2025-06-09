document.getElementById("checkBtn").addEventListener("click", () => {
  const text = document.getElementById("text").value;

  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text })
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById("status").innerText = `Prediction: ${data.label}`;
  });
});
