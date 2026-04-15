async function analyzeText() {

    const text = document.getElementById("textInput").value;
    const loader = document.getElementById("loader");
    const resultBox = document.getElementById("resultBox");

    if (text.trim() === "") {
        alert("Please enter text!");
        return;
    }

    loader.classList.remove("hidden");
    resultBox.classList.add("hidden");

    const response = await fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: text })
    });

    const data = await response.json();

    loader.classList.add("hidden");
    resultBox.classList.remove("hidden");

    document.getElementById("resultText").innerText = data.result;
    document.getElementById("confidenceText").innerText =
        "Confidence: " + data.confidence + "%";

    document.getElementById("confidenceFill").style.width =
        data.confidence + "%";

    gsap.from(".result", {
        opacity: 0,
        y: 20,
        duration: 0.6
    });
}