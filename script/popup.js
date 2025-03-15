document.getElementById("scrape").addEventListener("click", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.scripting.executeScript({
            target: { tabId: tabs[0].id },
            files: ["script/content.js"]
        });
    });
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "SCRAPED_DATA") {
        document.getElementById("para").innerText = message.data;
}});