// Simplified but comprehensive text extraction
try {
    // Find main content area
    const contentSelectors = [
        'article', '.article-body', '.post-content', '.entry-content',
        '.content', '.main-content', '[role="main"]', '.story-body'
    ];
    
    let container = null;
    for (const selector of contentSelectors) {
        container = document.querySelector(selector);
        if (container) break;
    }
    
    // Fallback to body
    if (!container) {
        container = document.body;
    }
    
    // Extract text elements
    const nodes = [...container.querySelectorAll("h1, h2, h3, p, div, span, li")];
    
    const data = nodes
        .map(el => el.innerText ? el.innerText.trim() : '')
        .filter(text => {
            // Basic filtering
            if (text.length < 50) return false;
            
            // Remove noise patterns
            const noisePatterns = [
                /^[\d\s\-\.]+$/,
                /(קרא עוד|לחץ כאן|מידע נוסף|פרסומת|ממומן|תגובות|שתף|לייק)/i,
                /^(צילום|תמונה|וידאו|גלריה):/i
            ];
            
            if (noisePatterns.some(pattern => pattern.test(text))) {
                return false;
            }
            
            // Check for substantial content
            const hebrewChars = (text.match(/[\u0590-\u05FF]/g) || []).length;
            const englishChars = (text.match(/[a-zA-Z]/g) || []).length;
            const totalLetters = hebrewChars + englishChars;
            
            return totalLetters > text.length * 0.3 && totalLetters > 20;
        })
        .map(text => text
            .replace(/\| ממומן|ממומן|Taboola|by Taboola|מידע נוסף|אולי יעניין אותך גם|קרא עוד|לחץ כאן|פרסומת/g, '')
            .replace(/\s+/g, ' ')
            .trim()
        )
        .filter(text => text.length > 50);
    
    // Remove duplicates
    const seen = new Set();
    const uniqueData = data.filter(text => {
        const key = text.substring(0, 50).toLowerCase();
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });
    
    // Send data
    chrome.runtime.sendMessage({ 
        type: "SCRAPED_DATA", 
        data: uniqueData
    });
    
} catch (error) {
    console.error('Content extraction error:', error);
    // Send error message
    chrome.runtime.sendMessage({ 
        type: "EXTRACTION_ERROR", 
        error: error.message 
    });
}
