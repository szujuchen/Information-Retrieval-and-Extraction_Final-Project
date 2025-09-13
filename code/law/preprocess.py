import os
import zipfile
from striprtf.striprtf import rtf_to_text

# Define input and output directories
zip_dir = "./"  # Directory containing the .zip files
output_dir = "./context"  # Directory to save the converted text files
os.makedirs(output_dir, exist_ok=True)

def extract_zip_and_rename(zip_path, extract_to, zip_name):
    """Extract a zip file to the specified directory."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract files and find the RTF
            for file_name in zip_ref.namelist():
                if file_name.endswith(".rtf"):
                    extracted_path = os.path.join(extract_to, f"{zip_name}.rtf")
                    with open(extracted_path, "wb") as out_file:
                        out_file.write(zip_ref.read(file_name))
                    print(f"Extracted and renamed: {file_name} -> {extracted_path}")
        print(f"Extracted: {zip_path}")
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")

def convert_rtf_to_text(rtf_path, output_path):
    """Convert an RTF file to plain text and save the result."""
    try:
        # Try to read with UTF-8 encoding
        with open(rtf_path, 'r', encoding='utf-8') as file:
            rtf_content = file.read()
    except UnicodeDecodeError:
        # Fallback to GBK encoding for Chinese RTF files
        with open(rtf_path, 'r', encoding='gbk') as file:
            rtf_content = file.read()
    
    # Convert RTF content to plain text
    plain_text = rtf_to_text(rtf_content)
    
    # Save plain text to the output file
    with open(output_path, 'w', encoding='utf-8') as out_file:
        out_file.write(plain_text)
    print(f"Converted: {rtf_path} -> {output_path}")

def process_zip_files(zip_directory, output_directory):
    """Process all zip files in a directory."""
    for zip_file in os.listdir(zip_directory):
        if zip_file.endswith(".zip"):
            zip_path = os.path.join(zip_directory, zip_file)
            zip_name = os.path.splitext(zip_file)[0]
            extract_dir = "./files"
            # Extract zip file
            extract_zip_and_rename(zip_path, extract_dir, zip_name)
            
    # Process extracted RTF files
    for file in os.listdir("./files"):
        if file.endswith(".rtf"):
            rtf_path = os.path.join("./files", file)
            txt_output_path = os.path.join(output_directory, f"{os.path.splitext(file)[0]}.txt")
            convert_rtf_to_text(rtf_path, txt_output_path)

# Run the processing function
process_zip_files(zip_dir, output_dir)
print(f"All files processed. Text files are saved in: {output_dir}")

import json
import os
import re

context_dir = 'context'
laws = os.listdir(context_dir)
print(len(laws))

law_name_pattern = r"法規名稱：(.+)"
article_pattern = r"第\s*(\d+|\d+-\d+)\s*條\n([\s\S]+?)(?=(第\s*(\d+|\d+-\d+)\s*條|$))"
# article_pattern = r"(第\s*(\d+(-\d+)*)\s*條)(.*?)(?=第\s*\d+\s*條|$)"

articles = []
def parse_context(law_name, article_number, article_context):
        if re.search(r'\d+\s', article_context):  
                split_context = re.split(r'(?=\d+\s)', article_context)
        else:
            split_context = [article_context]

        whole_context = ""
        for idx, context in enumerate(split_context, start=1):
            context = re.sub(r'^\d+\s*', '', context.strip()).replace('\n', ' ')
            context = re.sub(r'第\s*[一二三四五六七八九十]+\s*(章|節|編|目|款)\s*[^第]*', '', context)
            context = re.sub(r'\s*第\s*[一二三四五六七八九十]+\s*(章|節|編|目|款)', '', context)
            context = context.strip()
            
            if '（刪除）' in context:
                continue
            if context: 
                # if sub_num is not None and idx > 1:
                #     id = f"{law_name}第{article_number}-{sub_num}條之{idx-1}"
                # elif sub_num is not None:
                #     id = f"{law_name}第{article_number}-{sub_num}條"
                # elif idx > 1:
                #     id = f"{law_name}第{article_number}條之{idx-1}"
                # else:
                #     id = f"{law_name}第{article_number}條"
                whole_context += context
        if whole_context == "":
            return

        articles.append({
            "id": f"{law_name}第{article_number}條",
            "context": whole_context
        })


for lawfile in laws:
    with open(os.path.join(context_dir, lawfile), 'r', encoding='utf-8') as f:
        law_text = f.read()

    # law_name_match = re.search(law_name_pattern, law_text)
    # law_name = law_name_match.group(1) if law_name_match else "Unknown Law Name"
    law_name = lawfile.split('.')[0]

    article_matches = re.finditer(article_pattern, law_text)
    for match in article_matches:
        article_number = match.group(1)  
        article_context = match.group(2).strip()     
        parse_context(law_name, article_number, article_context) 
        
        
    
with open(f"law.json", "w") as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print(len(articles))
    

