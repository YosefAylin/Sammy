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
        var s = "";
        for(var i=0; i<message.data.length;i++){
            if(message.data[i].length > 75){
                s = s.concat(message.data[i]) + '\n';
            }
        }
        if(s == ""){
            alert("Nothing To Sammerize, Please Try A Different Page")
        }
        document.getElementById("para").innerText = s;
}});