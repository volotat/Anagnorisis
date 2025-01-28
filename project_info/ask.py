import os
import sys
import subprocess
import google.generativeai as genai

def get_codebase(input_dir):
    print("Collecting codebase content...")
    
    # Common text file extensions to include
    text_extensions = {
        '.py', '.js', '.java', '.c', '.cpp', '.h', '.hpp',
        '.html', '.css', '.scss', '.less', '.yaml', '.yml',
        '.txt', '.md', '.json', '.xml', '.csv', '.ini',
        '.cfg', '.conf', '.sh', '.bat', '.ps1', '.go',
        '.rs', '.php', '.rb', '.lua', '.sql', '.toml'
    }

    excluded_extensions = {'.min.js', '.min.css', '.map', '.csv',
        '.io.js', '.io.css', '.esm.js', '.esm.css', '.cjs.js', '.cjs.css',
    }

    excluded_files = { 'static/js/chart.js',
        'static/photoswipe/photoswipe.css'
    }

    def is_ignored(rel_path):
        """Check if a file is ignored by Git"""
        try:
            result = subprocess.run(
                ['git', 'check-ignore', '--quiet', rel_path],
                cwd=input_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Error checking Git ignore status for {rel_path}: {e}")
            return False

    # Collect all text files not ignored by Git
    files = []
    for root, dirs, filenames in os.walk(input_dir):
        # Include hidden directories (those starting with .)
        dirs[:] = [d for d in dirs if not d.startswith('.')]  # Remove this line to skip hidden directories
        
        for filename in filenames:
            # First check excluded full filename endings
            filename_lower = filename.lower()
            if any(filename_lower.endswith(ext) for ext in excluded_extensions):
                continue

            # Then check valid extensions
            ext = os.path.splitext(filename)[1].lower()
            if ext in text_extensions:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, input_dir)

                if rel_path in excluded_files:
                    continue   
                
                if not is_ignored(rel_path):
                    files.append((rel_path, full_path))

    # Build combined content string
    combined_content = []
    for rel_path, full_path in files:
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as infile:
                content = infile.read()
            
            combined_content.append(f"file: Anagnorisis/{rel_path}")
            combined_content.append("---- file start ----")
            combined_content.append(content)
            combined_content.append("---- file end ----\n")
        except Exception as e:
            print(f"Error processing {rel_path}: {str(e)}")

    final_content = "\n".join(combined_content)
    return final_content

# This function gets Git diff
def get_git_diff(input_dir):
    """Get current Git diff output"""
    try:
        # Get staged changes
        staged = subprocess.run(
            ['git', 'diff', '--staged'],
            cwd=input_dir,
            capture_output=True,
            text=True
        )
        # Get unstaged changes
        unstaged = subprocess.run(
            ['git', 'diff'],
            cwd=input_dir,
            capture_output=True,
            text=True
        )
        
        diff_output = ""
        if staged.stdout:
            diff_output += "STAGED CHANGES:\n" + staged.stdout + "\n"
        if unstaged.stdout:
            diff_output += "UNSTAGED CHANGES:\n" + unstaged.stdout + "\n"
            
        return diff_output.strip() or "No uncommitted changes detected."
        
    except Exception as e:
        print(f"Error getting Git diff: {str(e)}")
        return "Unable to retrieve Git diff information."


def main():
    # Configuration
    input_dir = os.path.abspath("../")
    system_prompt = """
    You are "Anagnorisis" project. 
    All the info about you as a project will be presented in a form of a current codebase. 
    Answer to each question as if you are talking about yourself. 

    If you need to analyze the code, carefully review the provided code files and provide detailed, professional responses.
    Consider best practices, potential issues, and optimization opportunities.
    Format your answers with clear headings and proper code formatting when needed."""
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        print("Please set it using:")
        print('export GEMINI_API_KEY="your-api-key"')
        sys.exit(1)
    
    genai.configure(api_key=api_key)

    # Create the model
    generation_config = {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # Initialize model with system instruction
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        system_instruction=system_prompt,
        generation_config=generation_config,
    )

    # Get codebase content
    codebase = get_codebase(input_dir)

    git_diff = get_git_diff(input_dir)

    # Get user question
    print("\nEnter your question about the codebase:")
    user_question = input("> ")

    # Prepare the full prompt
    full_prompt = f"""CODEBASE CONTEXT:
{codebase}

GIT DIFF CONTEXT (current uncommitted changes):
```diff
{git_diff}
```

USER QUESTION:
{user_question}"""

    # Save full prompt to file for debugging
    with open("llm_prompt.txt", "w") as f:
        f.write(full_prompt)
    
    # Generate and stream response
    print("\nGenerating response...\n")
    try:
        response = model.generate_content(full_prompt, stream=True)
        
        # Print streaming response
        full_response = []
        for chunk in response:
            chunk_text = chunk.text
            print(chunk_text, end="", flush=True)
            full_response.append(chunk_text)
        
        # Save full response to file
        with open("llm_response.md", "w") as f:
            f.write("".join(full_response))
            
    except Exception as e:
        print(f"\nError generating response: {str(e)}")

if __name__ == "__main__":
    main()