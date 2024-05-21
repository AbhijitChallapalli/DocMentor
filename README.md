# DocuMentor: PDF Text Explorer & Responder

## Description
This repository contains a Python application built with Streamlit that allows users to upload PDF files, extract text, and perform query answering using advanced language models from Hugging Face and vector storage using FAISS. It showcases the integration of various modern technologies in natural language processing and vector search.


## Usage

1. Clone this repository:
   
   git clone <repository-url>
   cd <repository-name>
   

2. Set up your environment variables:

   `OPENAI_API_KEY`: Your OpenAI API key for using models like GPT.
   `HUGGINGFACEHUB_API_TOKEN`: Your Hugging Face API token for accessing models on Hugging Face Hub.

   OPENAI_API_KEY=your_openai_api_key_here
   HUGGINGFACEHUB_API_TOKEN=your_huggingfacehub_api_token_here

   You can store these keys in a `.env` file in the root directory of your project:

   To run this project, you need to have Python installed on your machine (Python 3.8+ recommended). You can install the required dependencies with pip:

   pip install -r requirements.txt

3. Run the application:

   streamlit run main.py

4. This opens your web browser with a local host. Upload a PDF file and enter queries to interact with the text extracted from the PDF.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have suggestions for improvements or have identified bugs.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

Thanks to the developers and contributors of Streamlit, PyPDF2, OpenAI, and Hugging Face for making their tools available.
Special thanks to the LangChain community for their valuable tools and contributions to the language model integration.