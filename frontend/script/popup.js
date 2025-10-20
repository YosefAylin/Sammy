class SummaryUI {
    constructor() {
        this.paraElement = document.getElementById("para");
        this.scrapeButton = document.getElementById("scrape");
        this.lengthSlider = document.getElementById("length-slider");
        this.lengthValue = document.getElementById("length-value");
        this.lengthControl = document.getElementById("length-control");
        this.methodSelect = document.getElementById("method-select");
        this.copyButton = document.getElementById("copy-btn");
        this.currentSummary = "";
        this.lastProcessedText = "";
        this.lastTitle = "";
        this.currentUrl = "";
        
        this.initializeEventListeners();
        this.loadCachedSummary();
        this.loadSettings();
    }
    
    initializeEventListeners() {
        this.scrapeButton.addEventListener("click", () => this.handleScrape());
        
        if (this.lengthSlider) {
            this.lengthSlider.addEventListener("input", (e) => {
                this.updateLengthDisplay(e.target.value);
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
            // Always start with 15 for the first summary
            const length = 15; // Force default to קצר
            const method = result.summaryMethod || 'extractive';
            
            if (this.lengthSlider) {
                this.lengthSlider.value = length;
                this.updateLengthDisplay(length);
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
        
        // Hide slider when changing methods
        this.hideLengthControl();
        
        // Update button text based on method
        if (this.scrapeButton) {
            const buttonTexts = {
                'extractive': 'סכם דף (מהיר)',
                'abstractive': 'סכם דף (איכותי)'
            };
            this.scrapeButton.querySelector('.button-text').textContent = buttonTexts[method];
        }
    }
    
    updateLengthDisplay(value) {
        const numValue = parseInt(value);
        let displayText;
        
        if (numValue <= 15) {
            displayText = 'קצר';
        } else if (numValue <= 30) {
            displayText = 'רגיל';
        } else {
            displayText = 'ארוך';
        }
        
        if (this.lengthValue) {
            this.lengthValue.textContent = displayText;
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
        const isUltraShort = metadata.method === 'ultra_short_extractive';
        
        this.paraElement.innerHTML = `
            <div class="summary-container">
                <div class="summary-text">${data.summary}</div>
                ${(metadata.compression_ratio && metadata.summary_sentences && metadata.original_sentences) ? `
                    <div class="summary-stats">
                        דחיסה: ${Math.round(metadata.compression_ratio * 100)}% 
                        (${metadata.summary_sentences}/${metadata.original_sentences} משפטים)
                    </div>
                ` : ''}
                <div class="summary-actions">
                    <button id="copy-summary" class="action-btn">העתק</button>
                    ${(isUltraShort || metadata.summary_sentences <= 5) ? '<button id="expand-summary" class="action-btn expand-btn">הרחב סיכום</button>' : ''}
                    <button id="retry-summary" class="action-btn">סכם מחדש</button>
                </div>
            </div>
        `;
        
        // Add event listeners for new buttons
        document.getElementById("copy-summary").addEventListener("click", () => this.copyToClipboard());
        document.getElementById("retry-summary").addEventListener("click", () => this.handleScrape());
        
        // Show length control and add expand functionality for short summaries
        if (isUltraShort || metadata.summary_sentences <= 5) {
            this.showLengthControl();
            const expandBtn = document.getElementById("expand-summary");
            if (expandBtn) {
                expandBtn.addEventListener("click", () => this.expandSummary());
            }
        }
        
        // Save to cache
        this.saveSummaryToCache(data, this.lastProcessedText, this.lastTitle);
        
        this.hideLoading();
    }
    
    async handleScrape() {
        // Clear cache for this URL to force fresh summary
        if (this.currentUrl) {
            chrome.storage.session.remove([`summary_${this.currentUrl}`]);
        }
        
        this.showLoading();
        
        // Set a timeout for the entire operation
        const timeoutId = setTimeout(() => {
            this.showError("הפעולה נמשכת יותר מדי זמן - נסה שוב");
        }, 30000); // 30 second timeout
        
        try {
            const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
            
            // Execute content script
            await chrome.scripting.executeScript({
                target: { tabId: tabs[0].id },
                files: ["script/content.js"]
            });
            
            // Clear timeout if successful
            clearTimeout(timeoutId);
            
        } catch (error) {
            clearTimeout(timeoutId);
            console.error("Failed to execute content script:", error);
            
            if (error.message.includes('Cannot access')) {
                this.showError("לא ניתן לגשת לדף זה (דף מוגן או פנימי)");
            } else {
                this.showError("שגיאה בטעינת הדף - נסה לרענן ולנסות שוב");
            }
        }
    }
    
    async handleMessage(message, sender, sendResponse) {
        if (message.type === "SCRAPED_DATA") {
            const filteredText = this.processScrapedData(message.data);
            
            if (!filteredText) {
                this.showError("לא נמצא טקסט מתאים לסיכום");
                return;
            }
            
            // Include title in the request
            await this.requestSummary(filteredText, message.title);
        } else if (message.type === "EXTRACTION_ERROR") {
            console.error("Content extraction failed:", message.error);
            this.showError("שגיאה בחילוץ הטקסט מהדף");
        }
    }
    
    processScrapedData(data) {
        // Handle both array and string data
        if (typeof data === 'string') {
            return data.length > 100 ? data : null;
        }
        
        if (Array.isArray(data)) {
            const minLength = 75;
            const processedText = data
                .filter(text => text && text.length > minLength)
                .map(text => text.trim())
                .filter(text => text.length > minLength)
                .join(' ');
                
            return processedText.length > 100 ? processedText : null;
        }
        
        return null;
    }
    
    async requestSummary(text, title = '') {
        // Store for potential expansion
        this.lastProcessedText = text;
        this.lastTitle = title;
        
        const summaryPercentage = this.lengthSlider ? parseInt(this.lengthSlider.value) : 15;
        const method = this.methodSelect ? this.methodSelect.value : 'extractive';
        const startTime = performance.now();
        
        try {
            const payload = { 
                text: text,
                title: title,
                method: method,
                target_ratio: summaryPercentage / 100,  // Convert percentage to ratio
                max_length: Math.round(text.length * (summaryPercentage / 100) * 0.8)  // Estimate for abstractive
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
    
    showLengthControl() {
        if (this.lengthControl) {
            this.lengthControl.style.display = 'flex';
            // Add smooth animation
            this.lengthControl.style.opacity = '0';
            this.lengthControl.style.transform = 'translateY(-10px)';
            
            setTimeout(() => {
                this.lengthControl.style.transition = 'all 0.3s ease';
                this.lengthControl.style.opacity = '1';
                this.lengthControl.style.transform = 'translateY(0)';
            }, 100);
        }
    }
    
    hideLengthControl() {
        if (this.lengthControl) {
            this.lengthControl.style.display = 'none';
        }
    }
    
    async loadCachedSummary() {
        try {
            // Get current tab URL
            const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
            if (tabs.length === 0) return;
            
            this.currentUrl = tabs[0].url;
            
            // Check if we have a cached summary for this URL in session storage
            chrome.storage.session.get([`summary_${this.currentUrl}`], (result) => {
                const cachedData = result[`summary_${this.currentUrl}`];
                
                if (cachedData) {
                    // Show cached summary (no time check needed - session storage clears on tab close/refresh)
                    this.showCachedSummary(cachedData);
                }
            });
        } catch (error) {
            console.log("Could not load cached summary:", error);
        }
    }
    

    
    showCachedSummary(cachedData) {
        this.currentSummary = cachedData.summary;
        const metadata = cachedData.metadata || {};
        const isUltraShort = metadata.method === 'ultra_short_extractive';
        
        this.paraElement.innerHTML = `
            <div class="summary-container">
                <div class="cache-indicator">
                    📋 סיכום שמור (יימחק ברענון הדף)
                </div>
                <div class="summary-text">${cachedData.summary}</div>
                ${(metadata.compression_ratio && metadata.summary_sentences && metadata.original_sentences) ? `
                    <div class="summary-stats">
                        דחיסה: ${Math.round(metadata.compression_ratio * 100)}% 
                        (${metadata.summary_sentences}/${metadata.original_sentences} משפטים)
                    </div>
                ` : ''}
                <div class="summary-actions">
                    <button id="copy-summary" class="action-btn">העתק</button>
                    ${(isUltraShort || metadata.summary_sentences <= 5) ? '<button id="expand-summary" class="action-btn expand-btn">הרחב סיכום</button>' : ''}
                    <button id="refresh-summary" class="action-btn">סכם מחדש</button>
                </div>
            </div>
        `;
        
        // Add event listeners
        document.getElementById("copy-summary").addEventListener("click", () => this.copyToClipboard());
        document.getElementById("refresh-summary").addEventListener("click", () => this.handleScrape());
        
        // Show length control and add expand functionality for short summaries
        if (isUltraShort || metadata.summary_sentences <= 5) {
            this.showLengthControl();
            const expandBtn = document.getElementById("expand-summary");
            if (expandBtn) {
                expandBtn.addEventListener("click", () => this.expandSummary());
            }
        }
        
        // Store the cached data for potential expansion
        this.lastProcessedText = cachedData.originalText || "";
        this.lastTitle = cachedData.title || "";
    }
    
    saveSummaryToCache(summaryData, originalText, title) {
        if (!this.currentUrl) return;
        
        const cacheData = {
            summary: summaryData.summary,
            metadata: summaryData.metadata,
            originalText: originalText,
            title: title,
            timestamp: Date.now(),
            method: this.methodSelect ? this.methodSelect.value : 'extractive',
            length: this.lengthSlider ? parseInt(this.lengthSlider.value) : 15
        };
        
        // Save to Chrome session storage (clears on tab close/refresh)
        chrome.storage.session.set({
            [`summary_${this.currentUrl}`]: cacheData
        });
    }
    

    

    
    async expandSummary() {
        // Get current slider value for expansion
        const currentLength = parseInt(this.lengthSlider.value);
        const newLength = currentLength === 15 ? 30 : 45; // Move to next level
        
        this.lengthSlider.value = newLength;
        this.updateLengthDisplay(newLength);
        this.saveSettings();
        
        // Show loading and re-summarize with new length
        this.showLoading();
        
        // Get the last processed text (we'll need to store it)
        if (this.lastProcessedText) {
            await this.requestSummary(this.lastProcessedText, this.lastTitle);
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