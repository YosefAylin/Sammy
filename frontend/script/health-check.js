/**
 * Health Check for Sammy Extension
 * Monitors backend connectivity and model status
 */

class HealthChecker {
    constructor() {
        this.serverUrl = 'http://localhost:5002';
        this.checkInterval = 30000; // 30 seconds
        this.isChecking = false;
    }
    
    async checkServerHealth() {
        try {
            const response = await fetch(`${this.serverUrl}/health`, {
                method: 'GET',
                signal: AbortSignal.timeout(5000)
            });
            
            if (response.ok) {
                const data = await response.json();
                return {
                    status: 'healthy',
                    modelsLoaded: data.models_loaded,
                    device: data.device,
                    version: data.version
                };
            } else {
                return { status: 'error', message: `Server returned ${response.status}` };
            }
        } catch (error) {
            return { 
                status: 'offline', 
                message: error.name === 'TimeoutError' ? 'Server timeout' : 'Server offline'
            };
        }
    }
    
    async checkModelsStatus() {
        try {
            const response = await fetch(`${this.serverUrl}/models/status`, {
                method: 'GET',
                signal: AbortSignal.timeout(5000)
            });
            
            if (response.ok) {
                const data = await response.json();
                return data.models;
            }
        } catch (error) {
            return null;
        }
    }
    
    updateStatusIndicator(status) {
        const indicator = document.getElementById('status-indicator');
        if (!indicator) return;
        
        const statusConfig = {
            'healthy': { color: '#10b981', text: '●' },
            'error': { color: '#f59e0b', text: '●' },
            'offline': { color: '#ef4444', text: '●' }
        };
        
        const config = statusConfig[status] || statusConfig['offline'];
        indicator.style.color = config.color;
        indicator.textContent = config.text;
        indicator.title = `Server status: ${status}`;
    }
    
    async performHealthCheck() {
        if (this.isChecking) return;
        this.isChecking = true;
        
        try {
            const health = await this.checkServerHealth();
            this.updateStatusIndicator(health.status);
            
            // Store health info for popup
            chrome.storage.local.set({ 
                serverHealth: health,
                lastHealthCheck: Date.now()
            });
            
        } finally {
            this.isChecking = false;
        }
    }
    
    startMonitoring() {
        // Initial check
        this.performHealthCheck();
        
        // Periodic checks
        setInterval(() => {
            this.performHealthCheck();
        }, this.checkInterval);
    }
}

// Auto-start health monitoring when popup loads
if (typeof window !== 'undefined') {
    const healthChecker = new HealthChecker();
    document.addEventListener('DOMContentLoaded', () => {
        healthChecker.startMonitoring();
    });
}