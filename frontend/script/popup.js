class SummaryUI {
    constructor() {
        this.paraElement = document.getElementById("para");
        this.scrapeButton = document.getElementById("scrape");
        this.lengthSlider = document.getElementById("length-slider");
        this.lengthValue = document.getElementById("length-value");
        this.methodSelect = document.getElementById("method-select");
        this.copyButton = document.getElementById("copy-btn");
        this.currentSummary = "";
        
        this.initializeEventListeners();
        this.loadSettings();
    }
    
    initializeEventListeners() {
        this.scrapeButton.addEventListener("click", () => this.handleScrape());
        
        if (this.lengthSlider) {
            this.lengthSlider.addEventListener("input", (e) => {
                this.lengthValue.textContent = e.target.value;
                this.saveSettings();
            });
        }
        
        if (this.methodSelect) {
            this.methodSelect.addEventListener("change", () => {
                this.saveSettings();
                this.updateMethodDescription();
            });
        }
        
        if (this.copyButton) {
            this.copyButton.addEventListener("click", () => this.copyToClipboard());
        }
        
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sender, sendResponse);
        });
    }
    
    loadSettings() {
        chrome.storage.local.get(['summaryLength', 'summaryMethod'], (result) => {
            const length = result.summaryLength || 6;
            const method = result.summaryMethod || 'extractive';
            
            if (this.lengthSlider) {
                this.lengthSlider.value = length;
                this.lengthValue.textContent = length;
            }
            
            if (this.methodSelect) {
                this.methodSelect.value = method;
                this.updateMethodDescription();
            }
        });
    }
    
    saveSettings() {
        const length = parseInt(this.lengthSlider.value);
        const method = this.methodSelect ? this.methodSelect.value : 'extractive';
        chrome.storage.local.set({ 
            summaryLength: length,
            summaryMethod: method 
        });
    }
    
    updateMethodDescription() {
        const method = this.methodSelect.value;
        const descriptions = {
            'extractive': 'בוחר משפטים חשובים מהטקסט המקורי',
            'abstractive': 'יוצר סיכום חדש בשפה טבעית'
        };
        
        // Update button text based on method
        if (this.scrapeButton) {
            const buttonTexts = {
                'extractive': 'סכם דף (מהיר)',
                'abstractive': 'סכם דף (איכותי)'
            };
            this.scrapeButton.querySelector('.button-text').textContent = buttonTexts[method];
        }
    }
    
    showLoading() {
        this.paraElement.innerHTML = `
            <div class="loading-container">
                <div class="loading"></div>
                <span>מעבד טקסט...</span>
            </div>
        `;
        this.scrapeButton.disabled = true;
    }
    
    hideLoading() {
        this.scrapeButton.disabled = false;
    }
    
    showError(message, isRetryable = true) {
        this.paraElement.innerHTML = `
            <div class="error-message">
                <div class="error-icon">⚠️</div>
                <div class="error-text">${message}</div>
                <div class="error-actions">
                    ${isRetryable ? '<button class="retry-btn" onclick="location.reload()">נסה שוב</button>' : ''}
                    <button class="help-btn" onclick="window.open('https://github.com/your-repo/issues', '_blank')">דווח על בעיה</button>
                </div>
            </div>
        `;
        this.hideLoading();
    }
    
    showSummary(data) {
        this.currentSummary = data.summary;
        const metadata = data.metadata || {};
        
        this.paraElement.innerHTML = `
            <div class="summary-container">
                <div class="summary-text">${data.summary}</div>
                ${metadata.compression_ratio ? `
                    <div class="summary-stats">
                        דחיסה: ${Math.round(metadata.compression_ratio * 100)}% 
                        (${metadata.summary_sentences}/${metadata.original_sentences} משפטים)
                    </div>
                ` : ''}
                <div class="summary-actions">
                    <button id="copy-summary" class="action-btn">העתק</button>
                    <button id="retry-summary" class="action-btn">סכם מחדש</button>
                </div>
            </div>
        `;
        
        // Add event listeners for new buttons
        document.getElementById("copy-summary").addEventListener("click", () => this.copyToClipboard());
        document.getElementById("retry-summary").addEventListener("click", () => this.handleScrape());
        
        this.hideLoading();
    }
    
    async handleScrape() {
        this.showLoading();
        
        try {
            const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
            await chrome.scripting.executeScript({
                target: { tabId: tabs[0].id },
                files: ["script/content.js"]
            });
        } catch (error) {
            console.error("Failed to execute content script:", error);
            this.showError("לא ניתן לגשת לדף זה");
        }
    }
    
    async handleMessage(message, sender, sendResponse) {
        if (message.type === "SCRAPED_DATA") {
            const filteredText = this.processScrapedData(message.data);
            
            if (!filteredText) {
                this.showError("לא נמצא טקסט מתאים לסיכום");
                return;
            }
            
            await this.requestSummary(filteredText);
        }
    }
    
    processScrapedData(data) {
        // Enhanced text processing
        const minLength = 75;
        const processedText = data
            .filter(text => text.length > minLength)
            .map(text => text.trim())
            .filter(text => text.length > minLength)
            .join('\n');
            
        return processedText || null;
    }
    
    async requestSummary(text) {
        const summaryLength = this.lengthSlider ? parseInt(this.lengthSlider.value) : 6;
        const method = this.methodSelect ? this.methodSelect.value : 'extractive';
        const startTime = performance.now();
        
        try {
            const payload = { 
                text: text,
                method: method,
                top_n: summaryLength,
                max_length: summaryLength * 25  // Approximate for abstractive
            };
            
            const response = await fetch("http://localhost:5002/summarize", {
                method: "POST",
                headers: { 
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify(payload),
                signal: AbortSignal.timeout(60000)  // Modern timeout handling
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.showSummary(data);
            
        } catch (error) {
            console.error("Summarization error:", error);
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                this.showError("השרת לא זמין. האם הוא פועל?");
            } else {
                this.showError("שגיאה בסיכום הטקסט");
            }
        }
    }
    
    async copyToClipboard() {
        if (!this.currentSummary) return;
        
        try {
            await navigator.clipboard.writeText(this.currentSummary);
            // Show temporary success message
            const originalText = document.getElementById("copy-summary").textContent;
            document.getElementById("copy-summary").textContent = "הועתק!";
            setTimeout(() => {
                const btn = document.getElementById("copy-summary");
                if (btn) btn.textContent = originalText;
            }, 2000);
        } catch (error) {
            console.error("Failed to copy:", error);
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SummaryUI();
});