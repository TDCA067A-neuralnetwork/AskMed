const q = document.getElementById("question");
const ask = document.getElementById("ask");
const clear = document.getElementById("clear");
const statusBox = document.getElementById("status");
const answer = document.getElementById("answer");
const answerText = document.getElementById("answerText");

async function submitQuestion() {
  const question = q.value.trim();
  if (!question) {
    statusBox.textContent = "Please enter a question.";
    return;
  }

  ask.disabled = true;
  ask.textContent = "Checking...";
  answer.classList.add("hidden");
  statusBox.textContent = "Running medical-domain classification...";

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question})
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed.");

    if (data.allowed) {
      statusBox.textContent =
        `Medical query • classifier confidence: ${data.confidence}`;
    } else {
      statusBox.textContent = `Blocked as ${data.classification}.`;
    }

    answerText.textContent = data.answer;
    answer.classList.remove("hidden");
  } catch (e) {
    statusBox.textContent = "Request failed.";
    answerText.textContent = e.message;
    answer.classList.remove("hidden");
  } finally {
    ask.disabled = false;
    ask.textContent = "Ask Clinical AI";
  }
}

ask.addEventListener("click", submitQuestion);

q.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") submitQuestion();
});

clear.addEventListener("click", () => {
  q.value = "";
  statusBox.textContent = "";
  answerText.textContent = "";
  answer.classList.add("hidden");
});

document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    q.value = chip.textContent;
    submitQuestion();
  });
});
