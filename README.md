# LlamaIndex RAG Application

A Flask-based web application that allows users to perform RAG on files stored in the data directory.


## Description
This application creates a searchable index using LlamaIndex of the files stored in the data/ directory. A Kurt Vonnegut short story is already there for testing.

## Installation

1. Clone the repository and create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure
```
.
├── README.md
├── app.py
└── data/
    |── kurt.txt
```

## Usage

1. Start the server:
```bash
python app.py
```

2. The application will be available at `http://localhost:5000`

