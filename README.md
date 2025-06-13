## BoldChinese

An application for speech analysis and feedback designed to support Chinese language learners. It addresses the limitations of existing apps and offers a more effective learning experience.

## Overview

BoldChinese is an app that supports Chinese pronunciation practice. By leveraging speech recognition <br>technology and AI, it analyzes user pronunciation and provides immediate feedback.

## Key Features

Real-time voice analysis<br>
Scoring for pronunciation, intonation, and rhythm<br>
Detailed feedback and suggestions for improvement<br>
User-friendly interface
Technology Stack

## Frontend

Flutter/Dart<br>
FlutterFlow UI components<br>
Firebase Authentication

## Backend

Python (FastAPI)
Kaldi speech recognition engine
Custom kaidi library

## Backend Setup
```
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start backend
uvicorn app.main:app --reload

# Start frontend as well
cd BoldChinese
flutter run -d chrome
```
License

This project will be released under the MIT License.

## Work in process
Introducing the Workflow for Aligning Mandarin Speech using Kaidi<br>
https://qiita.com/reireu/items/53fa6b3dac3504d304ce<br>
2025-6-6<br>
<img width="644" alt="スクリーンショット 2025-06-06 11 33 38" src="https://github.com/user-attachments/assets/f5b763ac-8d19-4e14-ae56-7af7ee8b4cb8" />

