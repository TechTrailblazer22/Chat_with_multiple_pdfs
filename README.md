PDF Chatbot with LangChain and Hugging Face
This project is a conversational chatbot that allows you to upload PDF documents, extract their text, and ask questions based on the content. It uses LangChain for text processing and conversational memory, Hugging Face Transformers for embeddings and language models, and FAISS for efficient similarity search.

Features
Upload multiple PDF files.

Extract text from PDFs and split it into manageable chunks.

Use Hugging Face embeddings to create a vector store for semantic search.

Ask questions and get answers based on the content of the uploaded PDFs.

Conversational memory to maintain context across questions.

Requirements
To run this project, you need the following dependencies:

Python 3.8 or higher

Google Colab (optional, but recommended for easy setup)

When running the script in Google Colab, you will be prompted to upload PDF files. You can upload multiple files at once.

How It Works
Text Extraction:

The script extracts text from the uploaded PDFs using PyPDF2.

Text Chunking:

The extracted text is split into smaller chunks using CharacterTextSplitter from LangChain.

Embeddings and Vector Store:

The text chunks are converted into embeddings using Hugging Face's sentence-transformers/all-MiniLM-L6-v2 model.

A FAISS vector store is created to enable efficient similarity search.

Conversational Chain:

A conversational retrieval chain is set up using Hugging Face's airoboros-2.1-llama-2-13B-QLoRa model.

The chain maintains conversational memory to provide context-aware responses.

Chat Interface:

You can ask questions based on the content of the uploaded PDFs.

The chatbot provides answers and maintains a conversation history.

Usage
Running in Google Colab
Open the script in Google Colab.

Run the script and upload your PDF files when prompted.

Once the PDFs are processed, you can start asking questions.

Running Locally
Clone the repository or download the script.

Install the required dependencies (see Requirements).

Run the script:

bash
Copy
python pdf_chatbot.py
Follow the prompts to upload PDFs and ask questions.

Example
Uploading PDFs
plaintext
Copy
Upload your PDFs (you can select multiple files at once)
Asking Questions
plaintext
Copy
Ask a question (or type 'exit' to quit): What is the main topic of the document?
User: What is the main topic of the document?
Bot: The main topic of the document is...
Troubleshooting
1. TensorFlow Conflicts
If you encounter errors related to TensorFlow (e.g., RaggedTensorSpec), try the following:



2. Hugging Face API Key
Ensure your Hugging Face API key is correctly set in the .env file.

3. File Upload Issues
If you cannot upload multiple files in Colab, ensure you are using the latest version of the google-colab package.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
LangChain for providing the framework for text processing and conversational chains.

Hugging Face for pre-trained models and embeddings.

FAISS for efficient similarity search.

