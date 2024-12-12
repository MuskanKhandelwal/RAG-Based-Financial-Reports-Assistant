# Import Required Libraries
import os
import re
import json
from bs4 import BeautifulSoup
import pandas as pd
import PyPDF2


def process_file(uploaded_file, model_choice):
    # Detect file extension
    file_extension = uploaded_file.name.split(".")[-1]
    
    if file_extension == "pdf":
        # Extract text from PDF
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file_extension == "html":
        # Use the provided extract_text_from_html function
        content = uploaded_file.read().decode("utf-8")
        text = extract_text_from_html_stream(content)
    else:
        raise ValueError("Unsupported file format.")

    # Normalize and clean the text
    text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
    text = re.sub(r'\b(false|true|P1Y)\b', '', text, flags=re.IGNORECASE)  # Remove unwanted terms
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'(Item \d+[A-Z]?\.)', r'\n\1', text)  # Ensure headers start on new lines

    # Assume `cleaned_text` is the text you want to process
    cleaned_sections = process_text(text)
    mapped_data = map_headers_to_sections(cleaned_sections, header_mappings_10K)
    df = save_to_csv(mapped_data)
    print(df)

    return text

# Streamlined extract_text_from_html for Streamlit
def extract_text_from_html_stream(content):
    soup = BeautifulSoup(content, 'html.parser')
    raw_text = soup.get_text()
    return raw_text


# Clean extracted text
def clean_text(text):
    """
    Cleans the text by removing multiple spaces, newlines, tabs, and special characters.
    
    :param text: Raw text to clean.
    :return: Cleaned text.
    """
    # Remove multiple spaces, newlines, and tabs
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters (if needed)
    text = re.sub(r'[^\w\s.,]', '', text)
    # Strip leading/trailing whitespace
    return text.strip()

# Process raw text into structured sections
def process_text(cleaned_text):
    """
    Splits text into sections based on headers (Parts and Items).

    :param cleaned_text: The cleaned input text.
    :return: Dictionary of headers mapped to their corresponding sections.
    """
    # Define patterns for headers
    part_pattern = r"Part [IVXLCDM]+"
    item_pattern = r"Item \d+[A-Za-z]*\."
    # Combine patterns
    header_pattern = rf"{part_pattern}|{item_pattern}"

    # Split text into sections
    sections = re.split(header_pattern, cleaned_text, flags=re.IGNORECASE)

    # Find all headers
    headers = re.findall(header_pattern, cleaned_text, flags=re.IGNORECASE)

    # Map headers to their corresponding sections
    sections_dict = {header.strip(): content.strip() for header, content in zip(headers, sections[1:])}

    # Clean sections
    cleaned_sections = {header: clean_text(content) for header, content in sections_dict.items()}
    return cleaned_sections

# Map headers to cleaned sections
def map_headers_to_sections(cleaned_sections, header_mappings):
    """
    Maps the cleaned sections to predefined headers.

    :param cleaned_sections: Dictionary of cleaned sections.
    :param header_mappings: Mapping of original headers to friendly names.
    :return: Dictionary of friendly headers mapped to their sections.
    """
    return {column: cleaned_sections.get(item, "") for item, column in header_mappings.items()}

# Save mapped data to CSV
def save_to_csv(mapped_data, output_file="structured_10k.csv"):
    """
    Converts the mapped data into a single-row DataFrame and saves it as a CSV.

    :param mapped_data: Dictionary of mapped data to save.
    :param output_file: The output CSV file name.
    """
    # Convert to a single-row DataFrame
    df = pd.DataFrame([mapped_data])
    # Save to CSV
    df.to_csv(output_file, index=False)
    return df

# Header mappings
header_mappings_10K = {
    "Item 1.": "Business",
    "Item 1A.": "Risk Factors",
    "Item 1B.": "Unresolved Staff Comments",
    "Item 2.": "Properties",
    "Item 3.": "Legal Proceedings",
    "Item 4.": "Mine Safety Disclosures",
    "Item 5.": "Market for Registrant’s Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "Item 6.": "Selected Financial Data",
    "Item 7.": "Management’s Discussion and Analysis of Financial Condition and Results of Operations",
    "Item 7A.": "Quantitative and Qualitative Disclosures about Market Risk",
    "Item 8.": "Financial Statements and Supplementary Data",
    "Item 9.": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "Item 9A.": "Controls and Procedures",
    "Item 9B.": "Other Information",
    "Item 10.": "Directors, Executive Officers and Corporate Governance",
    "Item 11.": "Executive Compensation",
    "Item 12.": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "Item 13.": "Certain Relationships and Related Transactions, and Director Independence",
    "Item 14.": "Principal Accountant Fees and Services",
    "Item 15.": "Exhibits, Financial Statement Schedules"
}


