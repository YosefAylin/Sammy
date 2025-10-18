# 🤖 Sammy - AI-Powered Hebrew Summarizer

**Advanced Chrome extension for Hebrew text summarization using AlephBERT and mT5 models**

## 🎯 Project Overview

Sammy is an AI-powered browser extension that provides intelligent Hebrew text summarization using state-of-the-art NLP models. It offers both extractive and abstractive summarization methods optimized for Hebrew content.

## 🧠 AI Models Used

- **AlephBERT**: Hebrew-specific BERT model for semantic understanding (~500MB)
- **mT5**: Multilingual T5 for abstractive text generation (~1.2GB)

## 🌍 Cross-Platform Support

| Platform | Status | Python | Chrome |
|----------|--------|--------|--------|
| **Windows 10/11** | ✅ Fully Supported | 3.8+ | ✅ |
| **macOS** | ✅ Fully Supported | 3.8+ | ✅ |
| **Linux** | ✅ Fully Supported | 3.8+ | ✅ |

*See `CROSS_PLATFORM_GUIDE.md` for detailed compatibility info*

## 🚀 Quick Start (Universal)

### **Option 1: Virtual Environment (Recommended)**
```bash
git clone <repository-url>
cd sammy

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r backend/requirements.txt
pip install protobuf sentencepiece

# Start system
python run_sammy.py  # Auto-activates venv
```

### **Option 2: Automatic Setup**
```bash
git clone <repository-url>
cd sammy
python setup.py
```
*Handles everything automatically on Windows, macOS, and Linux*

### **Option 3: Manual Setup**
```bash
# 1. Install dependencies
pip install -r backend/requirements.txt
pip install protobuf sentencepiece

# 2. Download AI models (~1.6GB)
python preload_models.py

# 3. Start system
python start_sammy.py
```

### **Option 3: Step by Step**
```bash
python preload_models.py    # Download models
python backend/ai_server.py # Start AI server
```

### 4. Install Chrome Extension
1. Open `chrome://extensions/`
2. Enable Developer mode
3. Click "Load unpacked"
4. Select the `frontend` folder

## 📁 Project Structure

```
sammy/
├── 📁 backend/
│   ├── 🤖 ai_server.py           # Main AI server (CORE)
│   ├── 🔧 model_manager.py       # AI model management
│   ├── 📊 create_hebrew_dataset.py # Dataset creation tools
│   ├── 🎓 train_custom_model.py   # Model training utilities
│   └── 📋 requirements.txt        # Python dependencies
├── 📁 frontend/
│   ├── 🎨 css/style.css          # Modern Hebrew UI
│   ├── 📁 script/
│   │   ├── 🖱️ popup.js           # Extension popup logic
│   │   ├── 📄 content.js         # Web page content extraction
│   │   └── ⚙️ background.js      # Extension background service
│   ├── 🏠 index.html             # Extension popup interface
│   └── 📋 manifest.json          # Chrome extension configuration
├── 📁 models/                    # AI models (downloaded locally)
│   ├── 🧠 alephbert-base/        # Hebrew BERT (~500MB)
│   └── 🤖 mt5-small/             # Multilingual T5 (~1.1GB)
├── 🔧 preload_models.py          # AI model downloader
├── 🚀 start_sammy.py             # One-command startup
├── 🐍 run_sammy.py               # Virtual environment wrapper
├── 📁 venv/                      # Virtual environment (excluded from git)
├── 🔧 activate_env.sh            # Easy activation script
├── 📖 VIRTUAL_ENV_GUIDE.md       # Virtual environment documentation
└── 🧪 test_extension.html        # Testing page
```

## 📊 File Classification

### 🔴 **CORE FILES (Essential)**
| File | Purpose | Size | Critical |
|------|---------|------|----------|
| `backend/ai_server.py` | Main AI server with AlephBERT + mT5 | ~400 lines | ✅ YES |
| `frontend/script/popup.js` | Extension UI logic | ~200 lines | ✅ YES |
| `frontend/manifest.json` | Chrome extension config | ~50 lines | ✅ YES |
| `frontend/index.html` | Extension interface | ~80 lines | ✅ YES |
| `frontend/css/style.css` | Hebrew-optimized styling | ~300 lines | ✅ YES |
| `backend/requirements.txt` | Python dependencies | ~20 lines | ✅ YES |

### 🟡 **UTILITY FILES (Helpful)**
| File | Purpose | Expandable | Keep? |
|------|---------|------------|-------|
| `preload_models.py` | Downloads AI models | ✅ Yes | 🟡 Useful |
| `generate_icons.py` | Creates extension icons | ✅ Yes | 🟡 Useful |
| `test_extension.html` | Testing Hebrew content | ✅ Yes | 🟡 Useful |

### 🟢 **DEVELOPMENT FILES (Optional)**
| File | Purpose | Expandable | Keep? |
|------|---------|------------|-------|
| `backend/create_hebrew_dataset.py` | Dataset creation tools | ✅ Yes | 🟢 Optional |
| `backend/train_custom_model.py` | Model training utilities | ✅ Yes | 🟢 Optional |
| `frontend/script/content.js` | Web scraping logic | ✅ Yes | 🟢 Keep |
| `frontend/script/background.js` | Extension background | ✅ Yes | 🟢 Keep |

## 🎯 Core Functionality

### **AI Server (`backend/ai_server.py`)**
- **Purpose**: Main AI processing engine
- **Models**: AlephBERT for understanding + mT5 for generation
- **Methods**: 
  - Extractive: Selects important sentences using AI embeddings
  - Abstractive: Generates new summary text
- **Features**: Caching, batch processing, Hebrew optimization

### **Extension Frontend**
- **Popup**: Modern Hebrew interface with method selection
- **Content Script**: Intelligent Hebrew text extraction
- **Background**: Extension lifecycle management

### **AI Methods**
1. **Extractive (AlephBERT + TextRank)**:
   - Uses semantic embeddings for sentence scoring
   - Fast processing (~100ms)
   - High accuracy for Hebrew

2. **Abstractive (mT5)**:
   - Generates new summary text
   - Slower processing (~2s)
   - More natural language output

## 🔧 Technical Details

### **Dependencies**
- `torch` - PyTorch for AI models
- `transformers` - Hugging Face model library
- `flask` - Web server framework
- `numpy` - Numerical computations
- `scikit-learn` - ML utilities

### **Virtual Environment Setup**
This project uses Python virtual environments for:
- ✅ **Dependency isolation** - No conflicts with other projects
- ✅ **Reproducible builds** - Same environment everywhere
- ✅ **Cross-platform compatibility** - Works on Windows, macOS, Linux
- ✅ **Easy cleanup** - Delete `venv/` folder to remove everything

See `VIRTUAL_ENV_GUIDE.md` for detailed setup instructions.

### **Performance**
- **Memory**: ~2GB with both models loaded
- **Processing**: 100ms (extractive) / 2s (abstractive)
- **Models**: Cached after first load
- **Accuracy**: Optimized for Hebrew text patterns

## 🎨 Customization Options

### **Expandable Components**
1. **Add More Models**: Integrate Hebrew-specific models
2. **Enhanced UI**: Add more summarization options
3. **Better Training**: Fine-tune on Hebrew datasets
4. **Performance**: Add GPU acceleration
5. **Features**: Add translation, sentiment analysis

### **Configuration**
- Summary length: 3-10 sentences
- Processing method: Extractive vs Abstractive
- Model selection: Can switch between different AI models
- Hebrew optimization: Stopwords, connectors, linguistic rules

## 🚀 Production Ready

This implementation is production-ready with:
- ✅ Error handling and fallbacks
- ✅ Efficient caching and memory management
- ✅ Modern UI with Hebrew support
- ✅ Cross-browser compatibility
- ✅ Scalable architecture

## � FGit-Friendly Distribution

### **Why Models Aren't in Git**
- AI models are **~1.6GB total** (too large for Git)
- Each user downloads models individually to `./models/`
- Models are in `.gitignore` to keep repository lightweight

### **For Developers**
```bash
git clone <repository>
cd sammy
python3 preload_models.py  # Download models locally
python3 start_sammy.py     # Start the system
```

### **Repository Size**
- **Without models**: ~50MB (code only)
- **With models**: ~1.6GB (after download)
- **Git repository**: Stays lightweight for easy cloning

## 🔧 Production Features

- **Environment Configuration**: Configurable via environment variables
- **Performance Monitoring**: Built-in metrics and health checks
- **Error Handling**: Comprehensive error recovery and user feedback
- **Docker Support**: Containerized deployment ready
- **Optimized Dependencies**: Minimal production requirements
- **Health Monitoring**: Real-time server status in extension

## 📈 Future Enhancements

1. **Model Fine-tuning**: Train on Hebrew news datasets
2. **Performance**: GPU acceleration, model quantization  
3. **Features**: Multi-document summarization, key phrase extraction
4. **UI**: Dark mode, accessibility improvements
5. **Integration**: API for other applications
6. **Analytics**: Usage statistics and performance metrics

---

**Built with ❤️ for Hebrew NLP**