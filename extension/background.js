const SERVER_URL = "http://localhost:5001/log_tab";

function logActiveTab() {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (!tabs[0]) return;

        const tab = tabs[0];

        //skip internal chrome:// pages — nothing useful to log there
        if (!tab.url || tab.url.startsWith("chrome://")) return;

        fetch(SERVER_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                url: tab.url,
                title: tab.title,
                ts: Date.now()
            })
        }).catch(() => {
            //server probably not running yet — silently drop the event
        });
    });
}

chrome.tabs.onActivated.addListener(logActiveTab);
//fires when user switches to a different tab

chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
    if (changeInfo.status === "complete") logActiveTab();
    //fires when a tab finishes loading a new URL, not on every mid-load state change
});
