// Comprehensive text extraction following summarization guidelines
function extractComprehensiveContent() {
    // 1. Identify main content areas (opening, middle, closing)
    const contentSelectors = [
        'article', '.article-body', '.post-content', '.entry-content',
        '.content', '.main-content', '[role="main"]', '.story-body'
    ];
    
    let mainContent = null;
    for (const selector of contentSelectors) {
        mainContent = document.querySelector(selector);
        if (mainContent) break;
    }
    
    // Fallback to body if no main content found
    const container = mainContent || document.body;
    
    // 2. Extract all text elements with structure preservation
    const allNodes = [...container.querySelectorAll("h1, h2, h3, h4, h5, h6, p, div, span, li, td")];
    
    // 3. Process and categorize content by sections
    const sections = {
        opening: [],
        middle: [],
        closing: []
    };
    
    const processedTexts = allNodes
        .map(el => ({
            text: el.innerText.trim(),
            tag: el.tagName.toLowerCase(),
            position: Array.from(container.querySelectorAll('*')).indexOf(el)
        }))
        .filter(item => {
            // Filter meaningful content
            const text = item.text;
            return text.length > 30 && 
                   !isNoiseContent(text) &&
                   hasSubstantialContent(text);
        })
        .map(item => {
            // Clean and normalize text
            let cleanText = item.text
                .replace(/\| ממומן|ממומן|Taboola|by Taboola|מידע נוסף|אולי יעניין אותך גם|קרא עוד|לחץ כאן|פרסומת/g, '')
                .replace(/\s+/g, ' ')
                .trim();
            
            return { ...item, text: cleanText };
        })
        .filter(item => item.text.length > 30);
    
    // 4. Categorize into sections (opening, middle, closing)
    const totalItems = processedTexts.length;
    const openingEnd = Math.floor(totalItems * 0.25);
    const closingStart = Math.floor(totalItems * 0.75);
    
    processedTexts.forEach((item, index) => {
        if (index < openingEnd) {
            sections.opening.push(item.text);
        } else if (index >= closingStart) {
            sections.closing.push(item.text);
        } else {
            sections.middle.push(item.text);
        }
    });
    
    // 5. Remove duplicates while preserving section distribution
    const deduplicatedSections = {
        opening: removeDuplicates(sections.opening),
        middle: removeDuplicates(sections.middle),
        closing: removeDuplicates(sections.closing)
    };
    
    // 6. Combine all sections maintaining narrative flow
    const allContent = [
        ...deduplicatedSections.opening,
        ...deduplicatedSections.middle,
        ...deduplicatedSections.closing
    ];
    
    return {
        fullText: allContent.join(' '),
        sections: deduplicatedSections,
        metadata: {
            totalParagraphs: allContent.length,
            openingParagraphs: deduplicatedSections.opening.length,
            middleParagraphs: deduplicatedSections.middle.length,
            closingParagraphs: deduplicatedSections.closing.length,
            extractionMethod: 'comprehensive_sectional'
        }
    };
}

function isNoiseContent(text) {
    const noisePatterns = [
        /^[\d\s\-\.]+$/,  // Only numbers and punctuation
        /^[^\u0590-\u05FF\u0041-\u005A\u0061-\u007A]+$/,  // No Hebrew or English letters
        /(קרא עוד|לחץ כאן|מידע נוסף|פרסומת|ממומן|תגובות|שתף|לייק)/i,
        /^(צילום|תמונה|וידאו|גלריה):/i,
        /^[\W\s]*$/,  // Only whitespace and punctuation
        /^.{1,20}$/   // Too short to be meaningful
    ];
    
    return noisePatterns.some(pattern => pattern.test(text));
}

function hasSubstantialContent(text) {
    // Check for Hebrew content
    const hebrewChars = (text.match(/[\u0590-\u05FF]/g) || []).length;
    const englishChars = (text.match(/[a-zA-Z]/g) || []).length;
    const totalLetters = hebrewChars + englishChars;
    
    // Must have substantial letter content
    return totalLetters > text.length * 0.4 && totalLetters > 20;
}

function removeDuplicates(textArray) {
    const seen = new Set();
    return textArray.filter(text => {
        // Use first 50 characters as similarity key
        const key = text.substring(0, 50).toLowerCase();
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });
}

// Execute extraction and send to background
const extractedContent = extractComprehensiveContent();
chrome.runtime.sendMessage({ 
    type: "SCRAPED_DATA", 
    data: extractedContent.fullText,
    sections: extractedContent.sections,
    metadata: extractedContent.metadata
});
