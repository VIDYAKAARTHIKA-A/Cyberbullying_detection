chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "detectBullying",
    title: "Detect Cyberbullying",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "detectBullying") {
    const selectedText = info.selectionText;

    fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: selectedText })
    })
    .then(res => res.json())
    .then(data => {
      chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: (msg) => alert(msg),
        args: [`Result: ${data.label.toUpperCase()}`]
      });
    });
  }
});
