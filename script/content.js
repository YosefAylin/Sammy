// Example: Scrape all headings
var data = [...document.querySelectorAll("h1, h2, h3, p, span")].map(el => el.innerText).join("\n");

// Send data to background script
chrome.runtime.sendMessage({ type: "SCRAPED_DATA", data });
