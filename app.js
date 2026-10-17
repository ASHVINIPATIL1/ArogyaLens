const pages = [...document.querySelectorAll("[data-page]")];
const navLinks = [...document.querySelectorAll("[data-page-link]")];
const fileInput = document.querySelector("#file-input");
const messageInput = document.querySelector("#message-input");
const messageList = document.querySelector("#message-list");
const welcomePanel = document.querySelector("#welcome-panel");
const conversation = document.querySelector(".conversation");
const fileTray = document.querySelector("#file-tray");
const dropMessage = document.querySelector("#drop-message");
const themeToggle = document.querySelector("#theme-toggle");
const themeColorMeta = document.querySelector('meta[name="theme-color"]');
const THEME_STORAGE_KEY = "arogya-theme";
let selectedFile = null;
let currentSpeechButton = null;

// Keep the document theme, accessible toggle state, and browser chrome in sync.
function applyTheme(theme, { persist = false } = {}) {
  const isDark = theme === "dark";
  document.documentElement.dataset.theme = isDark ? "dark" : "light";
  themeToggle.setAttribute("aria-pressed", String(isDark));
  themeToggle.title = `Switch to ${isDark ? "light" : "dark"} mode`;
  themeToggle.setAttribute("aria-label", themeToggle.title);
  themeColorMeta.content = isDark ? "#131412" : "#fbf9f5";

  if (!persist) return;

  try {
    localStorage.setItem(THEME_STORAGE_KEY, isDark ? "dark" : "light");
  } catch (error) {
    console.warn("Theme changed for this session but could not be saved.", error);
  }
}

themeToggle.addEventListener("click", () => {
  const nextTheme =
    document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(nextTheme, { persist: true });
});

applyTheme(document.documentElement.dataset.theme === "dark" ? "dark" : "light");

// Page navigation
function navigateTo(pageName) {
  const page = pages.find((item) => item.dataset.page === pageName) ?? pages[0];
  pages.forEach((item) => {
    item.hidden = item !== page;
  });
  navLinks.forEach((link) => {
    const active = link.dataset.pageLink === page.dataset.page;
    link.classList.toggle("is-active", active);
    if (active) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });
  if (location.hash !== `#${page.dataset.page}`) {
    history.replaceState(null, "", `#${page.dataset.page}`);
  }
}

// Local document selection only; analysis is not connected in this prototype.
function showSelectedFile(file) {
  const validTypes = new Set([
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "image/png",
    "image/jpeg",
  ]);
  const validExtensions = /\.(pdf|docx|txt|png|jpe?g)$/i;

  if (!validTypes.has(file.type) && !validExtensions.test(file.name)) {
    dropMessage.textContent = "Choose a PDF, DOCX, TXT, JPG or PNG file.";
    return;
  }

  selectedFile = file;
  document.querySelector("#file-name").textContent = file.name;
  document.querySelector("#file-status").textContent = "Added locally · not yet analyzed";
  fileTray.hidden = false;
  dropMessage.textContent = "";
}

function clearSelectedFile() {
  selectedFile = null;
  fileInput.value = "";
  fileTray.hidden = true;
  dropMessage.textContent = "";
}

// Chat rendering
function addMessage(role, text) {
  const message = document.createElement("article");
  message.className = `message message-${role}`;

  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "ArogyaLens · prototype";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;

  message.append(label, bubble);

  if (role === "assistant") {
    const tools = document.createElement("div");
    tools.className = "message-tools";
    const speakButton = document.createElement("button");
    speakButton.className = "speech-button";
    speakButton.type = "button";
    speakButton.setAttribute("aria-label", "Read response aloud");
    speakButton.innerHTML = '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 10v4h4l5 4V6l-5 4H4Z" /><path d="M17 9a5 5 0 0 1 0 6m2.5-8.5a8.5 8.5 0 0 1 0 11" /></svg><span>Read aloud</span>';
    speakButton.addEventListener("click", () => {
      if (!("speechSynthesis" in window)) {
        dropMessage.textContent = "Read aloud isn’t available in this browser.";
        return;
      }

      if (currentSpeechButton === speakButton && window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
        currentSpeechButton = null;
        speakButton.querySelector("span").textContent = "Read aloud";
        speakButton.setAttribute("aria-label", "Read response aloud");
        return;
      }

      if (currentSpeechButton) {
        currentSpeechButton.querySelector("span").textContent = "Read aloud";
        currentSpeechButton.setAttribute("aria-label", "Read response aloud");
      }
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      currentSpeechButton = speakButton;
      speakButton.querySelector("span").textContent = "Stop reading";
      speakButton.setAttribute("aria-label", "Stop reading response");
      utterance.addEventListener("end", () => {
        if (currentSpeechButton === speakButton) {
          currentSpeechButton = null;
          speakButton.querySelector("span").textContent = "Read aloud";
          speakButton.setAttribute("aria-label", "Read response aloud");
        }
      });
      utterance.addEventListener("error", () => {
        if (currentSpeechButton === speakButton) {
          currentSpeechButton = null;
          speakButton.querySelector("span").textContent = "Read aloud";
          speakButton.setAttribute("aria-label", "Read response aloud");
          dropMessage.textContent = "The response could not be read aloud.";
        }
      });
      window.speechSynthesis.speak(utterance);
    });
    tools.append(speakButton);
    message.append(tools);
  }

  messageList.append(message);
}

function sendMessage(text) {
  const message = text.trim();
  if (!message) return;

  welcomePanel.hidden = true;
  messageList.classList.add("has-messages");
  addMessage("user", message);
  const documentNote = selectedFile
    ? ` I can see that you selected “${selectedFile.name}”, but this prototype does not read uploaded files.`
    : "";
  addMessage(
    "assistant",
    `Thanks for sharing your question. This is an early ArogyaLens prototype, so it can’t generate health answers or interpret documents yet.${documentNote} You can use this space to explore the interface; for personal medical guidance, please speak with a qualified healthcare professional.`,
  );
  document.querySelector("#recent-title").textContent =
    message.length > 28 ? `${message.slice(0, 28)}…` : message;
  messageInput.value = "";
  messageInput.style.height = "auto";
  dropMessage.textContent = "";
  messageList.lastElementChild?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

document.querySelector("#chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(messageInput.value);
});

messageInput.addEventListener("input", () => {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 132)}px`;
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage(messageInput.value);
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    messageInput.value = button.dataset.prompt;
    sendMessage(messageInput.value);
    messageInput.focus();
  });
});

document.querySelector("#upload-trigger").addEventListener("click", () => fileInput.click());
document.querySelector("#attach-trigger").addEventListener("click", () => fileInput.click());
document.querySelectorAll("[data-open-upload]").forEach((link) => {
  link.addEventListener("click", () => {
    window.setTimeout(() => fileInput.click(), 0);
  });
});

fileInput.addEventListener("change", () => {
  if (fileInput.files?.[0]) showSelectedFile(fileInput.files[0]);
});

document.querySelector("#remove-file").addEventListener("click", clearSelectedFile);

document.querySelector("#new-chat").addEventListener("click", () => {
  messageList.replaceChildren();
  messageList.classList.remove("has-messages");
  welcomePanel.hidden = false;
  document.querySelector("#recent-title").textContent = "A new conversation";
  clearSelectedFile();
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  currentSpeechButton = null;
  messageInput.value = "";
  navigateTo("chat");
  messageInput.focus();
});

document.querySelector("#recent-chat").addEventListener("click", () => {
  navigateTo("chat");
});

document.addEventListener("click", (event) => {
  const link = event.target.closest('a[href^="#"]');
  if (!link) return;
  const target = link.getAttribute("href").slice(1);
  if (target === "home" || target === "chat") {
    navigateTo(target);
  }
});

window.addEventListener("hashchange", () => {
  navigateTo(location.hash.slice(1) || "home");
});

["dragenter", "dragover"].forEach((eventName) => {
  conversation.addEventListener(eventName, (event) => {
    event.preventDefault();
    conversation.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  conversation.addEventListener(eventName, (event) => {
    event.preventDefault();
    conversation.classList.remove("is-dragging");
  });
});

conversation.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (file) showSelectedFile(file);
});

navigateTo(location.hash.slice(1) || "home");
