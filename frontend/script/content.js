// Balanced text extraction - main content without metadata
try {
    // Remove only clearly unwanted elements (ads, navigation, etc.)
    const unwantedSelectors = [
        // Ads and promotions
        '.ad', '.ads', '.advertisement', '.promo', '.sponsored', '.taboola', '.outbrain',
        '[class*="ad-"]', '[id*="ad-"]', '[class*="ads-"]', '[id*="ads-"]',

        // Navigation (but not content headers)
        'nav', '.navigation', '.menu', '.breadcrumb', '.pagination',

        // Comments and interactions
        '.comments', '.comment', '.discussion', '.feedback',

        // Popups and overlays
        '.popup', '.modal', '.overlay', '.newsletter-signup',

        // Hidden elements
        '[style*="display: none"]', '[style*="visibility: hidden"]', '.hidden'
    ];

    // Remove unwanted elements from the page
    unwantedSelectors.forEach(selector => {
        document.querySelectorAll(selector).forEach(el => el.remove());
    });

    // Find main content area with priority order - focus on article content only
    const contentSelectors = [
        'article',
        '.article-body', '.article-content', '.post-content', '.entry-content',
        '.story-body', '.story-content', '.main-content', '.content-body',
        '[role="main"]'
    ];

    let container = null;
    for (const selector of contentSelectors) {
        const element = document.querySelector(selector);
        if (element && element.innerText && element.innerText.length > 200) {
            // Additional check: make sure this container doesn't have betting content
            const containerText = element.innerText.toLowerCase();
            const hasBettingContent = /יתרון|רגיל|מעל\/מתחת|\d+\.\d+x\d+\.\d+/.test(containerText);
            
            if (!hasBettingContent) {
                container = element;
                break;
            }
        }
    }

    // Fallback: find the largest meaningful text container
    if (!container) {
        const candidates = [...document.querySelectorAll('div, section, article')];
        container = candidates
            .filter(el => {
                const text = el.innerText || '';
                const hebrewChars = (text.match(/[\u0590-\u05FF]/g) || []).length;
                return text.length > 200 && hebrewChars > text.length * 0.2;
            })
            .sort((a, b) => (b.innerText || '').length - (a.innerText || '').length)[0];
    }

    if (!container) {
        container = document.body;
    }

    // Extract only paragraphs and headings (main article content)
    const nodes = [...container.querySelectorAll("h1, h2, h3, h4, h5, h6, p")]
        .filter(el => {
            // Only visible elements with meaningful text
            const style = window.getComputedStyle(el);
            const text = el.innerText || '';
            
            // Skip if not visible
            if (style.display === 'none' || style.visibility === 'hidden' || text.trim().length < 20) {
                return false;
            }
            
            // Skip elements that contain betting/odds content
            const hasBettingContent = /יתרון|רגיל|מעל\/מתחת|\d+\.\d+x\d+\.\d+|מתוך \d+ משחקים/.test(text.toLowerCase());
            if (hasBettingContent) {
                return false;
            }
            
            // Skip elements with class names that suggest betting/ads
            const className = el.className.toLowerCase();
            const hasBettingClass = /bet|odds|gambling|widget|sidebar|related/.test(className);
            if (hasBettingClass) {
                return false;
            }
            
            return true;
        });

    const data = nodes
        .map(el => {
            const text = el.innerText ? el.innerText.trim() : '';
            return text;
        })
        .filter(text => {
            // Basic filtering - not too strict
            if (text.length < 50) return false;

            // Filter obvious noise and betting content
            const noisePatterns = [
                // Only numbers/symbols
                /^[\d\s\-\.,:\/]+$/,

                // Clear UI elements
                /^(קרא עוד|לחץ כאן|שתף|לייק|הירשם|תגובות)$/i,

                // External services
                /^(taboola|outbrain|sponsored|advertisement)$/i,

                // Email addresses and URLs
                /^[\w\.-]+@[\w\.-]+\.\w+$/,
                /^(www\.|http)/i,

                // Betting and odds content
                /(יתרון|רגיל|מעל\/מתחת|שערים|ניצחון|תיקו|הובסה)/i,
                /\d+\.\d+X\d+\.\d+/,  // Betting odds format like "2.80X2.80"
                /\(\d+\)\s*\d+:\d+/,   // Time format in betting
                /מתוך \d+ משחקים/,      // "Out of X games"
                
                // Team names that appear in betting content
                /(מכבי ראשון לציון|הפועל תל אביב|קרמונזה|אודינזה|ווסטהאם|ברנטפורד|ויסלה פלוק|טרמאליצה)/i,

                // Very short fragments
                /^.{1,20}$/
            ];

            if (noisePatterns.some(pattern => pattern.test(text))) {
                return false;
            }

            // Must have some Hebrew content (but not too strict)
            const hebrewChars = (text.match(/[\u0590-\u05FF]/g) || []).length;
            const englishChars = (text.match(/[a-zA-Z]/g) || []).length;
            const totalLetters = hebrewChars + englishChars;

            // Allow mixed content but require meaningful text
            return totalLetters > 20 && (hebrewChars > 10 || englishChars > 20);
        })
        .map(text => {
            // Light cleaning - only remove obvious noise
            return text
                .replace(/\bממומן\b|\bTaboola\b|\bby Taboola\b/g, '')
                .replace(/\s+/g, ' ')
                .trim();
        })
        .filter(text => text.length > 40);

    // Remove duplicates and very similar content
    const seen = new Set();
    const uniqueData = data.filter(text => {
        const key = text.substring(0, 80).toLowerCase();
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });

    // Final validation - ensure we have meaningful content
    const validData = uniqueData.filter(text => {
        // Must have reasonable length and word count
        const words = text.split(/\s+/).filter(w => w.length > 2);
        return words.length >= 8 && text.length >= 40;
    });

    // Extract page title for context
    const pageTitle = document.title || '';
    const h1Title = document.querySelector('h1')?.innerText || '';
    const mainTitle = h1Title || pageTitle;
    
    // Log for debugging
    console.log(`Extracted ${validData.length} valid text segments from ${nodes.length} elements`);
    console.log(`Page title: ${mainTitle}`);

    // Send data with title
    chrome.runtime.sendMessage({
        type: "SCRAPED_DATA",
        data: validData,
        title: mainTitle
    });

} catch (error) {
    console.error('Content extraction error:', error);
    // Send error message
    chrome.runtime.sendMessage({
        type: "EXTRACTION_ERROR",
        error: error.message
    });
}
