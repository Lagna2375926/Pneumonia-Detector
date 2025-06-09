// Navigation and page management
let currentPage = 'home';

function showPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    // Remove active class from all nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    
    // Show selected page
    const targetPage = document.getElementById(pageId);
    if (targetPage) {
        targetPage.classList.add('active');
        currentPage = pageId;
        
        // Add active class to corresponding nav link
        const navLink = document.querySelector(`a[href="#${pageId}"]`);
        if (navLink) {
            navLink.classList.add('active');
        }
    }
}

// Initialize navigation
document.addEventListener('DOMContentLoaded', function() {
    // Handle navigation clicks
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const pageId = this.getAttribute('href').substring(1);
            showPage(pageId);
        });
    });
    
    // Initialize file upload functionality
    initializeFileUpload();
    
    // Show home page by default
    showPage('home');
});

// File upload functionality
function initializeFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    if (!uploadArea || !fileInput) return;
    
    // Handle drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
    
    // Handle file input change
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
    
    // Handle click on upload area
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });
}

// Handle file upload and start analysis
function handleFileUpload(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file.');
        return;
    }
    
    // Show analysis section
    const analysisSection = document.getElementById('analysisSection');
    analysisSection.style.display = 'block';
    
    // Start analysis simulation
    startAnalysis('uploaded');
}

// Sample image analysis
function analyzeSample(type) {
    // Show analysis section
    const analysisSection = document.getElementById('analysisSection');
    analysisSection.style.display = 'block';
    
    // Start analysis with sample type
    startAnalysis(type);
}

// Analysis simulation - Fixed version
function startAnalysis(imageType) {
    const progressSection = document.getElementById('analysisProgress');
    const resultsSection = document.getElementById('analysisResults');
    const progressFill = document.getElementById('progressFill');
    
    // Show progress, hide results
    progressSection.style.display = 'block';
    resultsSection.style.display = 'none';
    
    // Reset progress
    progressFill.style.width = '0%';
    
    // Reset all steps
    document.querySelectorAll('.step').forEach(step => {
        step.classList.remove('active', 'completed');
    });
    
    // Simulate analysis steps with proper timing
    const steps = [
        { id: 'step1', text: 'Preprocessing image', duration: 1000, progress: 25 },
        { id: 'step2', text: 'Running AI models', duration: 1500, progress: 50 },
        { id: 'step3', text: 'Ensemble prediction', duration: 1000, progress: 75 },
        { id: 'step4', text: 'Generating results', duration: 800, progress: 100 }
    ];
    
    // Execute steps sequentially
    function executeSteps() {
        steps.forEach((step, index) => {
            setTimeout(() => {
                // Mark previous step as completed
                if (index > 0) {
                    const prevStep = document.getElementById(steps[index - 1].id);
                    if (prevStep) {
                        prevStep.classList.remove('active');
                        prevStep.classList.add('completed');
                    }
                }
                
                // Activate current step
                const stepElement = document.getElementById(step.id);
                if (stepElement) {
                    stepElement.classList.add('active');
                }
                
                // Update progress bar
                progressFill.style.width = step.progress + '%';
                
                // If this is the last step, show results after completion
                if (index === steps.length - 1) {
                    setTimeout(() => {
                        stepElement.classList.remove('active');
                        stepElement.classList.add('completed');
                        showResults(imageType);
                    }, step.duration);
                }
            }, steps.slice(0, index).reduce((acc, s) => acc + s.duration, 0));
        });
    }
    
    // Start the analysis
    executeSteps();
    
    // Scroll to analysis section
    document.getElementById('analysisSection').scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
    });
}

// Show analysis results
function showResults(imageType) {
    const progressSection = document.getElementById('analysisProgress');
    const resultsSection = document.getElementById('analysisResults');
    
    // Hide progress, show results
    progressSection.style.display = 'none';
    resultsSection.style.display = 'block';
    
    // Generate results based on image type
    let results;
    if (imageType === 'normal' || imageType === 'uploaded') {
        results = {
            diagnosis: 'Normal',
            confidence: Math.floor(Math.random() * 8) + 92, // 92-99%
            risk: 'low',
            riskText: 'Low Risk',
            recommendations: [
                'No immediate concerns detected',
                'Continue routine monitoring',
                'Consult physician for complete evaluation',
                'Regular follow-up as recommended'
            ],
            diagnosisClass: 'normal'
        };
    } else {
        results = {
            diagnosis: 'Pneumonia Detected',
            confidence: Math.floor(Math.random() * 10) + 88, // 88-97%
            risk: 'high',
            riskText: 'High Risk',
            recommendations: [
                'Immediate medical attention required',
                'Antibiotic treatment may be necessary',
                'Monitor respiratory symptoms closely',
                'Follow-up imaging recommended'
            ],
            diagnosisClass: 'pneumonia'
        };
    }
    
    // Update UI with results
    const diagnosisElement = document.getElementById('diagnosis');
    const confidenceElement = document.getElementById('confidence');
    const riskIndicator = document.getElementById('riskIndicator');
    const recommendationsList = document.getElementById('recommendations');
    
    if (diagnosisElement) {
        diagnosisElement.textContent = results.diagnosis;
        diagnosisElement.className = `diagnosis ${results.diagnosisClass}`;
    }
    
    if (confidenceElement) {
        confidenceElement.textContent = `Confidence: ${results.confidence}%`;
    }
    
    if (riskIndicator) {
        riskIndicator.innerHTML = `<div class="risk-level ${results.risk}">${results.riskText}</div>`;
    }
    
    if (recommendationsList) {
        recommendationsList.innerHTML = results.recommendations
            .map(rec => `<li>${rec}</li>`)
            .join('');
    }
    
    // Update heatmap based on diagnosis
    updateHeatmap(results.diagnosisClass);
    
    // Animate results appearance
    resultsSection.style.opacity = '0';
    resultsSection.style.transition = 'opacity 0.6s ease-in-out';
    setTimeout(() => {
        resultsSection.style.opacity = '1';
    }, 100);
}

// Update heatmap visualization
function updateHeatmap(diagnosisType) {
    const heatmapView = document.getElementById('heatmapView');
    if (!heatmapView) return;
    
    const overlay = heatmapView.querySelector('.attention-overlay');
    const description = heatmapView.querySelector('p');
    
    if (overlay && description) {
        if (diagnosisType === 'pneumonia') {
            overlay.style.background = 'radial-gradient(circle, rgba(239, 68, 68, 0.4), transparent 70%)';
            overlay.style.animation = 'pulse 1.5s infinite';
            description.textContent = 'High attention areas indicating potential pneumonia patterns';
        } else {
            overlay.style.background = 'radial-gradient(circle, rgba(16, 185, 129, 0.3), transparent 70%)';
            overlay.style.animation = 'pulse 3s infinite';
            description.textContent = 'Distributed attention pattern consistent with normal lung tissue';
        }
    }
}

// Performance metrics animation
function animateMetrics() {
    const metricValues = document.querySelectorAll('.metric-value');
    
    metricValues.forEach(metric => {
        const finalValue = metric.textContent;
        const isPercentage = finalValue.includes('%');
        const numericValue = parseFloat(finalValue.replace('%', ''));
        
        if (isNaN(numericValue)) return;
        
        let currentValue = 0;
        const increment = numericValue / 50; // 50 animation steps
        const duration = 2000; // 2 seconds
        const stepTime = duration / 50;
        
        metric.textContent = isPercentage ? '0%' : '0';
        
        const timer = setInterval(() => {
            currentValue += increment;
            if (currentValue >= numericValue) {
                currentValue = numericValue;
                clearInterval(timer);
            }
            
            if (finalValue.includes('<')) {
                metric.textContent = '<3s';
            } else {
                metric.textContent = isPercentage ? 
                    Math.round(currentValue) + '%' : 
                    Math.round(currentValue).toString();
            }
        }, stepTime);
    });
}

// Intersection Observer for animations
function initializeAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                if (entry.target.classList.contains('metrics-grid')) {
                    animateMetrics();
                }
                
                // Add fade-in animation to cards
                if (entry.target.classList.contains('feature-card') || 
                    entry.target.classList.contains('metric-card')) {
                    entry.target.style.opacity = '0';
                    entry.target.style.transform = 'translateY(20px)';
                    entry.target.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
                    
                    setTimeout(() => {
                        entry.target.style.opacity = '1';
                        entry.target.style.transform = 'translateY(0)';
                    }, 100);
                }
            }
        });
    }, { threshold: 0.2 });
    
    // Observe elements for animation
    document.querySelectorAll('.feature-card, .metric-card, .metrics-grid').forEach(el => {
        observer.observe(el);
    });
}

// Utility functions
function formatNumber(num) {
    return num.toLocaleString();
}

function generateRandomConfidence(min = 85, max = 99) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

// Enhanced mobile navigation
function addMobileNavigation() {
    const navbar = document.querySelector('.navbar');
    const navList = document.querySelector('.navbar-nav');
    
    if (!navbar || !navList) return;
    
    if (window.innerWidth <= 768) {
        // Create mobile menu toggle if it doesn't exist
        let mobileToggle = document.querySelector('.mobile-toggle');
        if (!mobileToggle) {
            mobileToggle = document.createElement('button');
            mobileToggle.className = 'mobile-toggle';
            mobileToggle.innerHTML = '☰';
            mobileToggle.style.cssText = `
                display: block;
                background: none;
                border: none;
                font-size: 1.5rem;
                color: var(--color-medical-primary);
                cursor: pointer;
                padding: 8px;
                position: absolute;
                right: 16px;
                top: 50%;
                transform: translateY(-50%);
            `;
            
            navbar.querySelector('.container').style.position = 'relative';
            navbar.querySelector('.container').appendChild(mobileToggle);
            
            // Initially hide nav list on mobile
            navList.style.display = 'none';
            
            mobileToggle.addEventListener('click', function() {
                const isVisible = navList.style.display === 'flex';
                navList.style.display = isVisible ? 'none' : 'flex';
                
                if (!isVisible) {
                    navList.style.position = 'absolute';
                    navList.style.top = '100%';
                    navList.style.left = '0';
                    navList.style.right = '0';
                    navList.style.backgroundColor = 'var(--color-medical-surface)';
                    navList.style.flexDirection = 'column';
                    navList.style.padding = '16px';
                    navList.style.boxShadow = 'var(--shadow-md)';
                    navList.style.zIndex = '1001';
                }
            });
            
            // Close mobile menu when clicking on nav links
            document.querySelectorAll('.nav-link').forEach(link => {
                link.addEventListener('click', function() {
                    if (window.innerWidth <= 768) {
                        navList.style.display = 'none';
                    }
                });
            });
        }
    } else {
        // Remove mobile toggle on desktop
        const mobileToggle = document.querySelector('.mobile-toggle');
        if (mobileToggle) {
            mobileToggle.remove();
        }
        // Reset nav list styles for desktop
        navList.style.display = 'flex';
        navList.style.position = 'static';
        navList.style.flexDirection = 'row';
        navList.style.backgroundColor = 'transparent';
        navList.style.padding = '0';
        navList.style.boxShadow = 'none';
    }
}

// Initialize all functionality when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize animations
    setTimeout(initializeAnimations, 500);
    
    // Add smooth scrolling to internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const target = document.getElementById(targetId);
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add loading states to buttons
    document.querySelectorAll('.btn').forEach(button => {
        button.addEventListener('click', function() {
            if (!this.disabled) {
                this.style.opacity = '0.7';
                setTimeout(() => {
                    this.style.opacity = '1';
                }, 200);
            }
        });
    });
    
    // Keyboard navigation support
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            // Close any open modals or reset upload area
            const uploadArea = document.getElementById('uploadArea');
            if (uploadArea) {
                uploadArea.classList.remove('dragover');
            }
            
            // Close mobile menu
            const navList = document.querySelector('.navbar-nav');
            if (navList && window.innerWidth <= 768) {
                navList.style.display = 'none';
            }
        }
    });
    
    // Add responsive navigation for mobile
    addMobileNavigation();
});

// Handle window resize
window.addEventListener('resize', function() {
    addMobileNavigation();
});

// Error handling for image loading
document.addEventListener('DOMContentLoaded', function() {
    // Wait a bit for images to load before setting up error handlers
    setTimeout(() => {
        document.querySelectorAll('img').forEach(img => {
            img.addEventListener('error', function() {
                // Create placeholder for failed images
                const placeholder = document.createElement('div');
                placeholder.style.cssText = `
                    width: 100%;
                    height: 200px;
                    background: var(--color-medical-bg);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: var(--color-medical-text-light);
                    border-radius: var(--radius-base);
                    border: 1px solid var(--color-border);
                    font-size: var(--font-size-sm);
                `;
                placeholder.textContent = 'Chart visualization';
                
                if (this.parentNode) {
                    this.parentNode.replaceChild(placeholder, this);
                }
            });
        });
    }, 1000);
});

// Analytics and tracking (placeholder for future implementation)
function trackUserInteraction(action, category, label) {
    // Placeholder for analytics tracking
    console.log(`Analytics: ${category} - ${action} - ${label}`);
}

// Accessibility improvements
document.addEventListener('DOMContentLoaded', function() {
    // Add ARIA labels to interactive elements
    document.querySelectorAll('.btn').forEach(button => {
        if (!button.getAttribute('aria-label')) {
            button.setAttribute('aria-label', button.textContent.trim());
        }
    });
    
    // Add skip navigation link
    const skipLink = document.createElement('a');
    skipLink.href = '#main-content';
    skipLink.textContent = 'Skip to main content';
    skipLink.className = 'sr-only';
    skipLink.style.cssText = `
        position: absolute;
        top: -40px;
        left: 6px;
        background: var(--color-medical-primary);
        color: white;
        padding: 8px;
        text-decoration: none;
        border-radius: 4px;
        z-index: 1000;
        transition: top 0.3s ease;
    `;
    skipLink.addEventListener('focus', function() {
        this.style.top = '6px';
    });
    skipLink.addEventListener('blur', function() {
        this.style.top = '-40px';
    });
    
    document.body.insertBefore(skipLink, document.body.firstChild);
    
    // Add main content ID to first page section
    const mainContent = document.getElementById('home');
    if (mainContent) {
        mainContent.setAttribute('id', 'main-content');
    }
});