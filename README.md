# 🤖 Sammy - Hebrew Text Summarizer

**Chrome extension that uses AI to summarize Hebrew websites**

## 🚀 Quick Start

```bash
git clone https://github.com/YosefAylin/Sammy.git
cd Sammy
python config/scripts/setup.py
```

Then install the Chrome extension from the `frontend` folder. That's it!

## What is Sammy?

Sammy is a Chrome browser extension that can read Hebrew websites and create smart summaries. Just click one button and get the main points of any Hebrew article!

## What I Built

### 🧠 **AI Brain (Backend)**
- **Smart AI Server** that understands Hebrew text
- **Two AI Models**: 
  - AlephBERT (understands Hebrew) 
  - mT5 (creates summaries)
- **Two Summary Types**:
  - **Fast**: Picks the best sentences from the original text
  - **Smart**: Creates new summary text in its own words

### 🎨 **Chrome Extension (Frontend)**
- **Simple Interface** with Hebrew support
- **Slider** to choose how many sentences you want (3-10)
- **Method Selector** to pick fast or smart summarization
- **One-Click Operation** - just press "סכם דף" (Summarize Page)

### 🔧 **Easy Setup System**
- **Virtual Environment** keeps everything organized
- **Automatic Installation** downloads AI models
- **Cross-Platform** works on Windows, Mac, and Linux

## How to Run Sammy on Your Computer

### 🚀 **Automatic Setup**

```bash
# 1. Get the code
git clone https://github.com/YosefAylin/Sammy.git
cd Sammy

# 2. Run the setup script (does everything automatically!)
python config/scripts/setup.py
```

**That's it!** The setup script will:
- ✅ Check your system compatibility
- ✅ Create virtual environment
- ✅ Install all Python packages
- ✅ Download AI models (~1.6GB)
- ✅ Test everything works
- ✅ Show you next steps

### 🎯 **Start Sammy**
```bash
# Option 1: Auto-activate virtual environment
python config/scripts/run_sammy.py

# Option 2: Manual (if venv already active)
python start_sammy.py
```
*Keep this running! You should see "Server running on: http://localhost:5002"*

### 📱 **Install Chrome Extension**
1. Open Chrome browser
2. Go to `chrome://extensions/`
3. Turn on "Developer mode" (top right toggle)
4. Click "Load unpacked"
5. Select the `frontend` folder from this project
6. Done! You'll see Sammy icon in your toolbar

## How to Use Sammy

1. **Go to any Hebrew website** (like Ynet, Haaretz, etc.)
2. **Click the Sammy icon** in your Chrome toolbar
3. **Choose your settings**:
   - Move slider for summary length (3-10 sentences)
   - Pick method: "חילוץ חכם" (fast) or "יצירה חדשה" (smart)
4. **Click "סכם דף"** (Summarize Page)
5. **Get your summary!** Copy it or try different settings

## Project Structure (What Each Folder Does)

```
Sammy/
├── backend/              # AI server and models
│   ├── ai_server.py     # Main AI brain
│   ├── requirements.txt # What Python packages we need
│   └── model_manager.py # Downloads and manages AI models
├── frontend/            # Chrome extension
│   ├── index.html      # Extension popup window
│   ├── script/         # JavaScript code
│   └── css/           # Styling
├── config/             # Configuration files
├── models/            # AI models (downloaded automatically)
└── venv/             # Python environment (created by you)
```

## Troubleshooting

### "Server not responding" error
- Make sure `python start_sammy.py` is running
- Check that you see "Server running on: http://localhost:5002"

### "Missing packages" error
- Make sure virtual environment is activated: `source venv/bin/activate`
- Reinstall requirements: `pip install -r backend/requirements.txt`

### Extension not working
- Reload the extension: Chrome → Extensions → Find Sammy → Click reload
- Make sure you selected the `frontend` folder, not the whole project

### Models not downloading
- Check internet connection
- Try running `python preload_models.py` again
- Models are ~1.6GB total, so it takes time

## What Makes This Special

### 🎯 **Smart Hebrew Processing**
- Understands Hebrew grammar and sentence structure
- Removes ads and noise from websites
- Keeps the most important information

### ⚡ **Two Speed Options**
- **Fast Mode**: Picks best sentences (100ms)
- **Smart Mode**: Creates new summary (2-3 seconds)

### 🎛️ **User Control**
- Choose exactly how many sentences you want
- Switch between different AI methods
- Works on any Hebrew website

### 🔧 **Professional Setup**
- Virtual environment keeps everything clean
- Easy installation process
- Works on Windows, Mac, and Linux

## For Developers

### Key Files to Understand
- `backend/ai_server.py` - Main AI logic
- `frontend/script/popup.js` - Extension interface
- `frontend/script/content.js` - Extracts text from websites
- `start_sammy.py` - Starts everything

### How It Works
1. Extension extracts text from Hebrew website
2. Sends text to AI server (localhost:5002)
3. AI server processes with AlephBERT + mT5 models
4. Returns summary to extension
5. User sees summary in popup

### Technologies Used
- **Python**: AI server and models
- **JavaScript**: Chrome extension
- **PyTorch**: AI model framework
- **Transformers**: Hugging Face AI library
- **Flask**: Web server for AI API

---

**Built by Yosef for Hebrew NLP** 🇮🇱