# Image analyzer for hotel room images
import os
import json
import base64
from pathlib import Path
from tqdm import tqdm
import requests
import time

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ImageAnalyzer:
    def __init__(self, api_key=None):
        """Initialize with OpenAI API key."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Provide it directly or via .env file.")
    
    def encode_image(self, image_path):
        """Encode image as base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def download_image(self, url, output_path, max_retries=3):
        """Download image from URL and save it to the specified path."""
        # Try with multiple URL formats in case one fails
        urls_to_try = [
            url,  # Original URL
            url.replace(".s3.eu-central-1.amazonaws.com", ""),  # Alternate format 1
            f"https://obilet.com/CaseStudy/HotelImages/{url.split('/')[-1]}"  # Alternate format 2
        ]
        
        for retry in range(max_retries):
            for current_url in urls_to_try:
                try:
                    # Disable SSL verification to handle certificate issues
                    response = requests.get(current_url, stream=True, verify=False)
                    response.raise_for_status()  # Raise exception for HTTP errors
                    
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    print(f"Successfully downloaded from {current_url}")
                    return True
                except Exception as e:
                    print(f"Error downloading image from {current_url}: {str(e)}")
                    # Continue to next URL or retry
            
            # If all URLs failed, wait before retrying
            if retry < max_retries - 1:
                time.sleep(1)  # Wait 1 second before retrying
        
        return False
    
    def process_urls_from_json(self, json_file, images_dir, desc_dir="descriptions"):
        """Download images from URLs in JSON file."""
        # Create images directory if it doesn't exist
        images_path = Path(images_dir)
        images_path.mkdir(exist_ok=True, parents=True)

        # Create descriptions directory if it doesn't exist
        desc_path = Path(desc_dir)
        desc_path.mkdir(exist_ok=True, parents=True)

        # Check for existing descriptions
        existing_descriptions = {}
        all_desc_path = desc_path / "all_descriptions.json"
        if all_desc_path.exists():
            try:
                with open(all_desc_path, 'r') as f:
                    existing_descriptions = json.load(f)
                print(f"Found {len(existing_descriptions)} existing descriptions.")
            except Exception as e:
                print(f"Error loading existing descriptions: {str(e)}")
        
        # Load the URLs from JSON file
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            if 'image_urls' not in data:
                raise ValueError("JSON file must contain 'image_urls' key with a list of URLs")
                
            urls = data['image_urls']
            
            # Download each image
            for i, url in enumerate(tqdm(urls, desc="Downloading images")):
                # Extract filename from URL or use index
                filename = url.split('/')[-1]
                img_num = Path(filename).stem  # Get the number part without extension
                output_path = images_path / filename
                
                # Skip if description already exists
                if img_num in existing_descriptions:
                    print(f"Skipping {filename} - description already exists")
                    continue
                
                # Skip if file already exists and has valid size
                if output_path.exists() and output_path.stat().st_size > 0:
                    print(f"Skipping download for {filename} - file already exists")
                    continue
                
                # Download the image
                success = self.download_image(url, output_path)
                if not success:
                    print(f"Failed to download image from {url} after multiple attempts")
                    # Create a placeholder file so we don't keep trying to download it
                    with open(output_path, 'w') as f:
                        f.write(f"Failed to download from {url}")
            
            return True
        except Exception as e:
            print(f"Error processing URLs from JSON: {str(e)}")
            return False
    
    def analyze_image(self, image_path):
        """Analyze the hotel room image using OpenAI's GPT-4.1-mini model."""
        try:
            # Check if the file is empty (placeholder for failed download)
            if Path(image_path).stat().st_size < 1000:  # Less than 1KB
                with open(image_path, 'r') as f:
                    content = f.read()
                if content.startswith("Failed to download"):
                    raise Exception(f"Skipping analysis - {content}")
            
            # Encode the image
            base64_image = self.encode_image(image_path)
            
            # Prepare the system prompt for image analysis with structured JSON output
            system_prompt = """
            You are a hotel room feature extractor. Analyze this hotel room image and provide a detailed description 
            in a structured JSON format with the following fields:
            
            {
                "room_type": "Single, Double, Triple, Suite, etc.",
                "max_capacity": "Maximum number of people the room can accommodate (integer)",
                "view_type": "Sea view, garden view, city view, mountain view, no specific view, etc.",
                "features": ["List of features: Desk, balcony, air conditioning, TV, minibar, etc."],
                "room_size": "Small, medium, large (approximate)",
                "bed_configuration": "Single beds, double beds, king size, etc.",
                "design_style": "Modern, classic, minimalist, etc.",
                "extra_amenities": ["List of notable amenities visible in the room"],
                "description": "A thorough and objective analysis of the room"
            }
            
            Provide a thorough and objective analysis focusing only on what is clearly visible in the image.
            Your response must be valid JSON that can be parsed directly. Do not include any text outside the JSON structure.
            """
            
            # Make API call to OpenAI
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please analyze this hotel room image and extract detailed features according to the categories provided."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 800
            }
            
            # Try up to 3 times with exponential backoff
            max_retries = 3
            for retry in range(max_retries):
                try:
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    
                    # Check if request was successful
                    if response.status_code == 200:
                        response_data = response.json()
                        json_content = response_data['choices'][0]['message']['content']
                        
                        # Try to parse the JSON response
                        try:
                            # Clean up the response if it contains markdown code blocks
                            if json_content.startswith('```json'):
                                json_content = json_content.split('```json')[1]
                            if '```' in json_content:
                                json_content = json_content.split('```')[0]
                                
                            # Parse the JSON content
                            structured_data = json.loads(json_content.strip())
                            
                            # Extract the description and add the structured data
                            description = structured_data.get('description', '')
                            structured_data['description'] = description
                            
                            # Set the structured data and description
                            break
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON response: {e}")
                            # Fallback to using the raw content as description
                            description = json_content
                            structured_data = {"description": description}
                            break
                    elif response.status_code == 429 or response.status_code >= 500:
                        # Rate limit or server error, retry after delay
                        wait_time = (2 ** retry) * 3  # Exponential backoff: 3, 6, 12 seconds
                        print(f"API rate limit or server error. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        if retry == max_retries - 1:
                            raise Exception(f"API request failed after {max_retries} retries: {response.text}")
                    else:
                        # Other error, don't retry
                        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
                except Exception as e:
                    if retry < max_retries - 1:
                        wait_time = (2 ** retry) * 3
                        print(f"Error during API call: {str(e)}. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise
            
            # Get the original URL for this image (from filename)
            filename = Path(image_path).name
            original_url = f"https://static.obilet.com.s3.eu-central-1.amazonaws.com/CaseStudy/HotelImages/{filename}"
            
            return {
                "image_path": str(image_path),
                "image_url": original_url,
                "description": description,
                **structured_data  # Include all structured data fields
            }
            
        except Exception as e:
            print(f"Error analyzing image {image_path}: {str(e)}")
            return {
                "image_path": str(image_path),
                "error": str(e)
            }
    
    def process_image_directory(self, image_dir, output_dir):
        """Process all images in a directory and save their descriptions to JSON files."""
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Check for existing descriptions
        existing_descriptions = {}
        all_desc_path = output_path / "all_descriptions.json"
        if all_desc_path.exists():
            try:
                with open(all_desc_path, 'r') as f:
                    existing_descriptions = json.load(f)
                print(f"Found {len(existing_descriptions)} existing descriptions.")
            except Exception as e:
                print(f"Error loading existing descriptions: {str(e)}")
        
        # Get all image files
        image_path = Path(image_dir)
        image_files = []
        
        # Get all image files with different extensions
        image_files.extend(list(image_path.glob("*.jpg")))
        image_files.extend(list(image_path.glob("*.jpeg")))
        image_files.extend(list(image_path.glob("*.png")))
        
        # Sort files by filename
        image_files.sort(key=lambda x: x.name)
        
        if not image_files:
            print(f"No image files found in {image_dir}")
            return existing_descriptions  # Return existing descriptions if no new images
        
        results = existing_descriptions.copy()  # Start with existing descriptions
        
        # Process each image
        for img_file in tqdm(image_files, desc="Analyzing images"):
            # Extract image number from filename (assuming format like "1.jpg")
            img_num = img_file.stem
            
            # Skip if already processed
            if img_num in results and "error" not in results[img_num]:
                print(f"Skipping {img_file.name} - already processed")
                continue
            
            # Analyze the image
            result = self.analyze_image(img_file)
            
            # Store the result
            results[img_num] = result
            
            # Save individual result to JSON
            with open(output_path / f"{img_num}.json", "w") as f:
                json.dump(result, f, indent=2)
            
            # Update the all_descriptions.json file after each image to save progress
            with open(output_path / "all_descriptions.json", "w") as f:
                json.dump(results, f, indent=2)
            
            # Short pause to avoid hitting API rate limits
            time.sleep(0.5)
        
        return results

# Example usage if this file is run directly
if __name__ == "__main__":
    try:
        # Suppress SSL warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        analyzer = ImageAnalyzer()
        # First download images from URLs in JSON file
        analyzer.process_urls_from_json("data/hotel_images.json", "images", "descriptions")
        # Then process the downloaded images
        results = analyzer.process_image_directory("images", "descriptions")
        print(f"Successfully analyzed {len(results)} images.")
    except Exception as e:
        print(f"An error occurred: {str(e)}") 